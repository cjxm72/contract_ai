import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, List

import requests
from loguru import logger

from util.PandocSettings import openai_settings
from model_manager import QwenModel


class ContractManager:
    def __init__(self):
        self.local_model = None
        self.model_mode = "online"
        self.temperature = 0.3
        self.max_tokens = openai_settings.max_tokens or 4048
        self.timeout = openai_settings.timeout_seconds if openai_settings.use_local_model else openai_settings.timeout
        self.max_rounds = 3  # 业务规则：最多3轮补充
        self.http_session = requests.Session()

        self._initialize_model_pipeline()
        self._load_contract_templates()

    def _initialize_model_pipeline(self):
        """根据配置加载本地模型或准备在线配置"""
        self.temperature = openai_settings.temperature if hasattr(openai_settings, "temperature") else self.temperature
        self.max_tokens = openai_settings.max_tokens or self.max_tokens

        if openai_settings.use_local_model:
            self.local_model = QwenModel()
            if self.local_model.load_model():
                self.model_mode = "local"
                logger.info("✅ 使用本地Qwen模型进行推理")
                return
            logger.warning("⚠️ 本地模型加载失败，切换为在线API")

        self.model_mode = "online"

    def _call_model(self, prompt: str, task: str) -> str:
        """根据当前模式调用模型，失败后自动重试或降级"""
        last_error = None
        if self.model_mode == "local" and self.local_model and self.local_model.is_loaded():
            try:
                return self._call_local_model(prompt)
            except Exception as e:
                last_error = e
                logger.warning(f"⚠️ 本地模型执行{task}失败，降级到在线API: {e}")
                self.model_mode = "online"

        try:
            return self._call_online_model(prompt)
        except Exception as e:
            if last_error:
                logger.error(f"❌ 本地与在线模型均失败，本地错误: {last_error}")
            raise e

    def _call_local_model(self, prompt: str) -> str:
        """调用本地Qwen模型"""
        if not self.local_model or not self.local_model.is_loaded():
            raise RuntimeError("本地模型未加载")

        model, tokenizer = self.local_model.get_model()
        if model is None or tokenizer is None:
            raise RuntimeError("本地模型组件未准备好")

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        output_ids = model.generate(
            **inputs,
            max_new_tokens=min(self.max_tokens, 1024),
            temperature=self.temperature,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if decoded.startswith(prompt):
            decoded = decoded[len(prompt):]
        return decoded.strip()

    def _call_online_model(self, prompt: str) -> str:
        """调用在线API模型"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_settings.api_key}"
        }
        payload = {
            "model": openai_settings.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": min(self.max_tokens, 4048),
            "temperature": self.temperature
        }

        retries = openai_settings.max_retries if hasattr(openai_settings, "max_retries") else 1

        for attempt in range(retries + 1):
            try:
                response = self.http_session.post(
                    f"{openai_settings.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"].strip()
                logger.warning(f"⚠️ 在线模型请求失败({attempt + 1}/{retries + 1}): {response.status_code} {response.text}")
            except Exception as exc:
                logger.warning(f"⚠️ 在线模型请求异常({attempt + 1}/{retries + 1}): {exc}")

        raise RuntimeError("在线模型请求失败，已达到最大重试次数")

    async def process_contract_interactive(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """交互式处理合同请求（语义分析 → 多轮补充 → 合同生成）"""
        try:
            conversation_history: List[Dict[str, str]] = []
            user_notes: List[str] = []
            asked_questions: List[str] = []  # 记录已提出的问题，避免重复
            info_complete = False
            analysis_prompt = self._prepare_initial_context(input_data, "", asked_questions, is_first_round=True)

            # 第0轮：接受JSON，AI分析并提问（必须输出）
            print(f"\n{'=' * 60}")
            print("🤖 AI正在执行语义级完整性分析...")
            print(f"{'=' * 60}")
            print(f"\n🔄 第 0 轮：初始分析")
            
            ai_response = await asyncio.to_thread(self._call_model, analysis_prompt, "analysis")
            ai_response = self._limit_text(ai_response, 300)
            info_complete = False  # 第0轮强制不完整，必须提问

            # 提取本次提出的问题，加入记忆
            new_questions = self._extract_questions(ai_response)
            asked_questions.extend(new_questions)

            conversation_history.append({
                "role": "assistant",
                "content": ai_response,
                "name": "SemanticReviewer"
            })

            # 第0轮必须输出
            print(f"\n{'=' * 60}")
            print("🤖 AI分析结果：")
            print(ai_response)
            print(f"{'=' * 60}")

            # 第0轮用户必须回答
            print("\n💬 请根据以上问题补充信息（多行输入，空行结束，输入exit可终止）：")
            user_input = self._get_user_input().strip()

            if not user_input:
                logger.warning("⚠️ 用户未提供补充信息，将基于现有信息生成")
            elif user_input.lower() in {"exit", "quit", "q", "停止", "退出"}:
                return {
                    "status": "error",
                    "message": "用户终止对话",
                    "conversation": conversation_history
                }
            else:
                user_notes.append(user_input)
                conversation_history.append({
                    "role": "user",
                    "content": user_input,
                    "name": "User"
                })

            # 第1、2、3轮：基于用户回答继续分析（只有不完整才输出）
            for round_idx in range(1, self.max_rounds + 1):  # round_idx: 1, 2, 3
                # 构建分析提示
                merged_supplements = "\n".join(f"- {note}" for note in user_notes)
                analysis_prompt = self._prepare_initial_context(input_data, merged_supplements, asked_questions, is_first_round=False)
                
                print(f"\n🔄 第 {round_idx} 轮：基于用户回答继续分析")
                ai_response = await asyncio.to_thread(self._call_model, analysis_prompt, "analysis")
                ai_response = self._limit_text(ai_response, 300)
                info_complete = self._should_stop_questioning(ai_response)

                # 提取本次提出的问题，加入记忆
                new_questions = self._extract_questions(ai_response)
                asked_questions.extend(new_questions)

                conversation_history.append({
                    "role": "assistant",
                    "content": ai_response,
                    "name": "SemanticReviewer"
                })

                # 只有判断为不完整时才输出到终端
                if not info_complete:
                    print(f"\n{'=' * 60}")
                    print("🤖 AI分析结果：")
                    print(ai_response)
                    print(f"{'=' * 60}")

                    # 如果是第3轮（round_idx == 3），用户回答后直接生成，不再审查
                    if round_idx == self.max_rounds:
                        print("\n⚠️ 已达到最大对话轮次，将基于现有信息生成合同范本")
                        break

                    # 第1、2轮继续提问
                    print("\n💬 请根据以上问题补充信息（多行输入，空行结束，输入exit可终止）：")
                    user_input = self._get_user_input().strip()

                    if not user_input:
                        logger.warning("⚠️ 用户未提供补充信息，将基于现有信息生成")
                        break

                    if user_input.lower() in {"exit", "quit", "q", "停止", "退出"}:
                        return {
                            "status": "error",
                            "message": "用户终止对话",
                            "conversation": conversation_history
                        }

                    user_notes.append(user_input)
                    conversation_history.append({
                        "role": "user",
                        "content": user_input,
                        "name": "User"
                    })
                else:
                    # 信息完整，直接生成
                    print(f"\n{'=' * 60}")
                    print("✅ 信息完整，开始生成合同")
                    print("[DEBUG] 🤖 AI分析结果：")
                    print(ai_response)
                    print(f"{'=' * 60}")
                    break

            # 无论信息是否完整，都生成合同范本
            if info_complete:
                print("\n🎯 信息完整，开始生成合同范本...")
            else:
                print("\n🎯 基于现有信息生成合同范本（请注意：此范本需要进一步修改完善）...")

            final_data = input_data.copy()
            if user_notes:
                final_data["user_supplements"] = "\n\n".join(user_notes)

            generation_prompt = self._build_generation_prompt(final_data)
            contract_content = await asyncio.to_thread(self._call_model, generation_prompt, "generation")

            conversation_history.append({
                "role": "assistant",
                "content": contract_content,
                "name": "ContractGenerator"
            })

            return {
                "status": "success",
                "contract": contract_content,
                "conversation": conversation_history,
                "message": "合同生成成功"
            }

        except Exception as e:
            logger.error(f"❌ 合同处理失败: {e}")
            raise

    def _get_user_input(self) -> str:
        """获取用户多行输入（空行结束）"""
        lines = []
        print("请输入内容（空行结束）：")
        while True:
            try:
                line = input()
                if not line.strip():  # 空行结束
                    break
                lines.append(line)
            except EOFError:
                break
        return "\n".join(lines)

    def _limit_text(self, text: str, max_chars: int) -> str:
        """限制模型输出长度，超出部分截断并标记"""
        if not text or len(text) <= max_chars:
            return text
        logger.warning(f"⚠️ 模型输出长度超过{max_chars}字，将自动截断")
        return text[:max_chars] + "..."

    def _should_stop_questioning(self, reviewer_message: str) -> bool:
        """判断是否应该停止提问"""
        if not reviewer_message:
            return False
        status = self._extract_status_tag(reviewer_message)
        if status == "完整":
            return True
        if status == "不完整":
            return False

        normalized = reviewer_message.replace(" ", "")
        if "不完整" in normalized or "无法生成" in normalized or "需补充" in normalized:
            return False
        if "信息完整" in normalized and "不完整" not in normalized:
            return True
        if "可以生成合同" in normalized and "不可以" not in normalized:
            return True
        return False

    def _extract_status_tag(self, reviewer_message: str) -> str:
        """解析模型输出中的状态标签"""
        if not reviewer_message:
            return ""
        match = re.search(r"【状态】\s*(完整|不完整)", reviewer_message)
        if match:
            return match.group(1)
        return ""

    def _extract_questions(self, ai_response: str) -> List[str]:
        """从AI回答中提取提出的问题，用于记忆避免重复"""
        questions = []
        if not ai_response:
            return questions

        # 尝试从【问题】标签中提取
        question_match = re.search(r"【问题】\s*([^\n]+(?:\n[^\n]+)*)", ai_response)
        if question_match:
            question_text = question_match.group(1).strip()
            # 按行分割，每行作为一个问题
            for line in question_text.split('\n'):
                line = line.strip()
                if line and line != "无" and len(line) > 5:  # 过滤太短的内容
                    questions.append(line)
        else:
            # 如果没有【问题】标签，尝试提取问号结尾的句子
            question_sentences = re.findall(r'[^。！？]*[？?][^。！？]*', ai_response)
            for q in question_sentences:
                q = q.strip()
                if q and len(q) > 5:
                    questions.append(q)

        return questions

    def _prepare_initial_context(self, input_data: Dict[str, Any], supplements: str = "", asked_questions: List[str] = None, is_first_round: bool = False) -> str:
        """准备语义分析上下文"""
        if asked_questions is None:
            asked_questions = []

        context = """请分析以下合同信息的完整性，识别缺失或不明确的信息。注意：生成的合同仅为范本，需要人工进一步修改完善，因此有点严格但不必过于严格。

    【合同基本信息】
    """

        for key, value in input_data.items():
            if value:
                context += f"- {key}: {value}\n"

        if supplements:
            context += f"""
【用户最新补充】
{supplements}
"""

        if asked_questions:
            context += f"""
【已提出的问题（请勿重复提问）】
"""
            for i, q in enumerate(asked_questions, 1):
                context += f"{i}. {q}\n"

        # 第一轮特殊提示：必须反问
        if is_first_round:
            context += """
    【第一轮分析要求（重要）】
    这是第一轮分析，必须找出至少1-3个问题向用户提问，以提升用户体验，整体字数不得超过300字。
    重点关注：
    1. JSON中未提及的关键信息（如：具体交付标准、验收标准、保密条款等）
    2. 信息不明确的地方（如：金额单位、时间节点、责任划分等）
    3. 即使信息看起来完整，也要找出可以进一步明确或补充的点
    4. 输出必须严格遵循以下格式（禁止添加多余段落）：
       【状态】完整/不完整（仅可二选一）可以生成合同必须显示完整
       【结论】一句话概括结果，若完整需包含"可以生成合同"
       【问题】若状态为不完整，列出数条待补充问题；若状态为完整，填写"无"
    除非信息真的非常完整且无任何可优化空间，否则必须提问。
    """

        context += """
    请按照以下要求分析并输出，整体字数不得超过300字：
    1. 判断关键信息是否基本完善，注意这是合同范本，不需要100%完整
    2. 若存在缺失或不明确信息，仅列出最关键的数条问题（避免重复已问过的问题）
    3. 输出必须严格遵循以下格式（禁止添加多余段落）：
       【状态】完整/不完整（仅可二选一）可以生成合同必须显示完整
       【结论】一句话概括结果，若完整需包含"可以生成合同"
       【问题】若状态为不完整，列出数条待补充问题；若状态为完整，填写"无"
    4. 要求关键要素（主体、服务内容、金额/支付、期限等）基本明确"
    5. 不允许输出json的字段，即键值对的键，涉及也不允许！！！！
    6. 严禁重复提出已列在"已提出的问题"中的相同或类似问题"""

        if is_first_round:
            context += """
    7. 【第一轮强制要求】必须找出至少1-3个问题，除非信息真的非常完整且无任何可优化空间"""

        context += """

    请开始分析："""

        return context

    def _build_generation_prompt(self, input_data: Dict[str, Any]) -> str:
        """构建合同生成提示词"""
        contract_type = input_data.get('contract_type', '').lower()
        template_key = self._select_template(contract_type)
        template_content = self.templates.get(template_key, "")

        prompt = f"""请基于以下完整信息生成一份专业的合同范本：

【合同信息】
{json.dumps(input_data, ensure_ascii=False, indent=2)}"""

        if template_content:
            prompt += f"""

【参考模板格式】
{template_content}

**重要**：参考模板的结构和条款类型，但不要直接复制模板内容。基于提供的具体信息生成全新的合同内容。"""

        prompt += """

**生成要求**：
1. 在合同开头必须添加以下声明（使用醒目标记）：
   "---
   【重要提示】本合同由AI自动生成，仅为参考范本，必须经过专业法律人员审核和修改后方可使用。不得直接使用，否则可能产生法律风险。
   ---"
2. 基于提供的具体信息生成合同内容
3. 条款完整、合法、专业
4. 使用规范的合同语言
5. 包含必要的合同要素
6. 输出纯合同内容（包含开头的声明），不要包含额外说明

请生成合同："""

        return prompt

    def _select_template(self, contract_type: str) -> str:
        """选择合同模板"""
        for key in self.templates.keys():
            if key in contract_type or contract_type in key:
                return key
        return 'default'

    def _load_contract_templates(self):
        """加载合同模板 - 找不到模板直接报错"""
        # 初始化模板目录
        current_dir = Path(__file__).parent
        self.template_dir = current_dir.parent / "templates"
        self.templates = {}

        logger.info(f"📁 查找模板目录: {self.template_dir}")

        if not self.template_dir.exists():
            error_msg = f"❌ 模板目录不存在: {self.template_dir}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        template_files = list(self.template_dir.glob("*.md"))

        if not template_files:
            error_msg = f"❌ 模板目录中没有找到任何.md模板文件: {self.template_dir}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info(f"🔍 找到 {len(template_files)} 个模板文件")

        # 加载所有模板文件
        for template_file in template_files:
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    self.templates[template_file.stem] = f.read()
                logger.info(f"✅ 加载模板: {template_file.name}")
            except Exception as e:
                error_msg = f"❌ 加载模板失败 {template_file.name}: {e}"
                logger.error(error_msg)
                raise Exception(error_msg)

        logger.info(f"📊 成功加载 {len(self.templates)} 个模板: {list(self.templates.keys())}")



    def run(self):
        """运行演示主函数（同步入口，内部调用异步逻辑）"""
        # 用asyncio.run()包裹异步核心逻辑
        asyncio.run(self._async_run())

    async def _async_run(self):
        """异步核心主函数（交互式处理模式）"""
        try:
            logger.info("🚀 启动合同生成演示程序")
            print("=" * 70)
            print("🤖 智能合同生成系统（交互式模式）")
            print("=" * 70)

            # 设置路径
            current_dir = Path(__file__).parent
            input_dir = current_dir.parent / "input"

            logger.info(f"🔍 查找input目录: {input_dir}")
            print(f"🔍 查找input目录: {input_dir}")

            if not input_dir.exists():
                logger.error(f"❌ input目录不存在: {input_dir}")
                print(f"❌ 错误: input目录不存在")
                return

            json_files = list(input_dir.glob("*.json"))
            if not json_files:
                logger.warning(f"⚠️ input目录中没有JSON文件: {input_dir}")
                print(f"❌ 错误: input目录中没有找到JSON文件")
                return

            print(f"📁 找到 {len(json_files)} 个合同文件:")
            for i, json_file in enumerate(json_files, 1):
                print(f"  {i}. {json_file.name}")

            # 交互式处理：逐个文件处理
            print(f"\n{'=' * 70}")
            print("🔍 交互式处理模式")
            print(f"{'=' * 70}")

            for i, json_file in enumerate(json_files, 1):
                print(f"\n📄 处理文件 {i}/{len(json_files)}: {json_file.name}")

                with open(json_file, 'r', encoding='utf-8') as f:
                    input_data = json.load(f)

                print(f"📋 初始信息:")
                for key, value in input_data.items():
                    print(f"  - {key}: {value}")

                # 异步方法加await
                result = await self.process_contract_interactive(input_data)

                if result['status'] == 'success':
                    output_dir = current_dir.parent / "output"
                    output_dir.mkdir(exist_ok=True)
                    output_file = output_dir / f"{json_file.stem}.md"

                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(result['contract'])

                    print(f"✅ 合同已保存: {output_file}")
                    print(f"📄 合同预览:\n{result['contract'][:300]}...")
                else:
                    print(f"❌ 生成失败: {result.get('message', '未知错误')}")

                if i < len(json_files):
                    continue_choice = input(f"\n继续处理下一个文件? (y/n): ").strip().lower()
                    if continue_choice != 'y':
                        break

            print(f"\n{'=' * 70}")
            print("🎉 程序执行完成")
            print(f"{'=' * 70}")

        except Exception as e:
            logger.error(f"❌ 演示程序执行失败: {e}")
            print(f"❌ 程序执行失败: {e}")
        finally:
            self.release_resources()

    def release_resources(self):
        """释放模型资源"""
        if self.local_model and self.local_model.is_loaded():
            self.local_model.release_resources()
            logger.info("✅ 已释放本地模型资源")
        if hasattr(self, "http_session") and self.http_session:
            try:
                self.http_session.close()
                logger.debug("🧹 HTTP会话已关闭")
            except Exception:
                pass

    def __del__(self):
        """析构函数"""
        self.release_resources()


if __name__ == "__main__":
    # 配置详细日志
    logger.remove()
    logger.add(
        "contract_system.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {message}",
        rotation="10 MB",
        retention="7 days"
    )
    logger.add(
        sys.stdout,
        level="DEBUG",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>"
    )

    # 运行演示程序
    manager = ContractManager()
    manager.run()