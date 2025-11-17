import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict as TypedDictExt

try:
    from langchain.memory import ConversationBufferMemory
except ImportError:
    # 如果导入失败，使用备用方案
    from langchain_community.chat_message_histories import ChatMessageHistory
    ConversationBufferMemory = None

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from loguru import logger

from util.PandocSettings import openai_settings
from model_manager import QwenModel


class ContractState(TypedDictExt):
    """合同处理状态"""
    input_data: Dict[str, Any]
    conversation_history: List[Dict[str, str]]
    user_notes: List[str]
    asked_questions: List[str]
    round_idx: int
    info_complete: bool
    contract_content: str
    template_key: Optional[str]
    max_rounds: int


class LangChainContractManager:
    """基于LangChain的合同管理器"""
    
    def __init__(self, template_key: Optional[str] = None):
        self.local_model = None
        self.model_mode = "online"
        self.temperature = 0.3
        self.max_tokens = openai_settings.max_tokens or 4048
        self.timeout = openai_settings.timeout_seconds if openai_settings.use_local_model else openai_settings.timeout
        self.max_rounds = 3
        self.selected_template_key = template_key
        # 初始化记忆（如果ConversationBufferMemory可用）
        if ConversationBufferMemory is not None:
            self.memory = ConversationBufferMemory(
                return_messages=True,
                memory_key="chat_history"
            )
        else:
            self.memory = None
        
        self._initialize_model_pipeline()
        self._load_contract_templates()
        self._build_workflows()
    
    def _initialize_model_pipeline(self):
        """根据配置加载本地模型或准备在线配置"""
        self.temperature = openai_settings.temperature if hasattr(openai_settings, "temperature") else self.temperature
        self.max_tokens = openai_settings.max_tokens or self.max_tokens

        if openai_settings.use_local_model:
            self.local_model = QwenModel()
            if self.local_model.load_model():
                self.model_mode = "local"
                logger.info("✅ 使用本地Qwen模型进行推理")
                # 创建本地模型包装器
                self.llm = self._create_local_llm()
                return
            logger.warning("⚠️ 本地模型加载失败，切换为在线API")
            self.model_mode = "online"

        # 创建在线模型
        self.llm = ChatOpenAI(
            model=openai_settings.model,
            api_key=openai_settings.api_key,
            base_url=openai_settings.base_url,
            temperature=self.temperature,
            max_tokens=min(self.max_tokens, 4048),
            timeout=self.timeout,
            max_retries=openai_settings.max_retries if hasattr(openai_settings, "max_retries") else 1,
        )
        self.model_mode = "online"
    
    def _create_local_llm(self):
        """创建本地模型包装器（兼容LangChain接口）"""
        from langchain_core.language_models import BaseChatModel
        from langchain_core.outputs import ChatGeneration, ChatResult
        from langchain_core.messages import AIMessage
        
        class LocalQwenLLM(BaseChatModel):
            """本地Qwen模型包装器"""
            
            def __init__(self, model, tokenizer, temperature, max_tokens):
                super().__init__()
                self.model = model
                self.tokenizer = tokenizer
                self.temperature = temperature
                self.max_tokens = max_tokens
            
            @property
            def _llm_type(self) -> str:
                return "local_qwen"
            
            def _generate(
                self,
                messages,
                stop=None,
                run_manager=None,
                **kwargs
            ):
                # 将消息转换为提示词
                prompt = self._messages_to_prompt(messages)
                
                inputs = self.tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=min(self.max_tokens, 1024),
                    temperature=self.temperature,
                    do_sample=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                if decoded.startswith(prompt):
                    decoded = decoded[len(prompt):]
                
                message = AIMessage(content=decoded.strip())
                generation = ChatGeneration(message=message)
                return ChatResult(generations=[generation])
            
            def _messages_to_prompt(self, messages):
                """将LangChain消息转换为提示词"""
                prompt = ""
                for msg in messages:
                    if isinstance(msg, SystemMessage):
                        prompt += f"System: {msg.content}\n"
                    elif isinstance(msg, HumanMessage):
                        prompt += f"User: {msg.content}\n"
                    elif isinstance(msg, AIMessage):
                        prompt += f"Assistant: {msg.content}\n"
                prompt += "Assistant: "
                return prompt
        
        model, tokenizer = self.local_model.get_model()
        return LocalQwenLLM(model, tokenizer, self.temperature, self.max_tokens)

    def _load_contract_templates(self):
        """加载合同模板"""
        current_dir = Path(__file__).parent
        self.template_dir = current_dir.parent / "templates"
        self.templates = {}

        logger.info(f"📁 查找模板目录: {self.template_dir}")

        if not self.template_dir.exists():
            error_msg = f"❌ 模板目录不存在: {self.template_dir}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # 修改：同时支持 .md 和 .docx 文件
        md_files = list(self.template_dir.glob("*.md"))
        docx_files = list(self.template_dir.glob("*.docx"))
        template_files = md_files + docx_files

        if not template_files:
            error_msg = f"❌ 模板目录中没有找到任何模板文件(.md或.docx): {self.template_dir}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info(f"🔍 找到 {len(template_files)} 个模板文件")

        for template_file in template_files:
            try:
                if template_file.suffix == '.md':
                    with open(template_file, 'r', encoding='utf-8') as f:
                        self.templates[template_file.stem] = f.read()
                elif template_file.suffix == '.docx':
                    # 需要安装 python-docx 库: pip install python-docx
                    from docx import Document
                    doc = Document(template_file)
                    content = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
                    self.templates[template_file.stem] = content
                logger.info(f"✅ 加载模板: {template_file.name}")
            except Exception as e:
                error_msg = f"❌ 加载模板失败 {template_file.name}: {e}"
                logger.error(error_msg)
                raise Exception(error_msg)

    def _select_template(self, contract_type: str = "", template_key: Optional[str] = None) -> str:
        """选择合同模板，支持外部参数指定"""
        # 优先使用构造函数传入的模板
        if self.selected_template_key and self.selected_template_key in self.templates:
            logger.info(f"✅ 使用主函数指定的模板: {self.selected_template_key}")
            return self.selected_template_key

        # 如果提供了外部参数，优先使用
        if template_key and template_key in self.templates:
            logger.info(f"✅ 使用指定的模板: {template_key}")
            return template_key

        # 如果没有提供或提供的模板不存在，则自主选择
        if contract_type:
            contract_type_lower = contract_type.lower()
            for key in self.templates.keys():
                if key.lower() in contract_type_lower or contract_type_lower in key.lower():
                    logger.info(f"✅ 自动选择模板: {key}")
                    return key

        # 默认返回第一个模板，如果没有则返回default
        if self.templates:
            default_key = list(self.templates.keys())[0]
            logger.info(f"✅ 使用默认模板: {default_key}")
            return default_key

        logger.warning("⚠️ 未找到可用模板")
        return 'default'
    
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
        """从AI回答中提取提出的问题"""
        questions = []
        if not ai_response:
            return questions

        question_match = re.search(r"【问题】\s*([^\n]+(?:\n[^\n]+)*)", ai_response)
        if question_match:
            question_text = question_match.group(1).strip()
            for line in question_text.split('\n'):
                line = line.strip()
                if line and line != "无" and len(line) > 5:
                    questions.append(line)
        else:
            question_sentences = re.findall(r'[^。！？]*[？?][^。！？]*', ai_response)
            for q in question_sentences:
                q = q.strip()
                if q and len(q) > 5:
                    questions.append(q)

        return questions
    
    def _check_contract_length(self, contract_content: str, max_length: int = 50000) -> bool:
        """检查合同长度，超过限制返回True"""
        return len(contract_content) > max_length
    
    def _split_contract_generation(self, prompt: str, max_chunk_size: int = 3000) -> List[str]:
        """分段生成合同，返回各段内容"""
        # 将提示词分成多个部分，分别生成
        # 这里简化处理，实际可以根据合同结构分段
        chunks = []
        # 可以按段落或章节分割提示词
        # 暂时返回单个提示词，由生成函数处理分段
        return [prompt]
    
    def _merge_contract_segments(self, segments: List[str]) -> str:
        """合并合同分段"""
        # 添加AI生成声明
        header = """---
【重要提示】本合同由AI自动生成，仅为参考范本，必须经过专业法律人员审核和修改后方可使用。不得直接使用，否则可能产生法律风险。
---

"""
        return header + "\n\n".join(segments)
    
    async def _generate_contract_segment(self, segment_prompt: str) -> str:
        """生成合同的一个分段"""
        try:
            response = await self.llm.ainvoke([HumanMessage(content=segment_prompt)])
            return response.content.strip()
        except Exception as e:
            logger.error(f"❌ 合同分段生成失败: {e}")
            raise
    
    def _build_workflows(self):
        """构建LangGraph工作流"""
        # 审查交互工作流
        self.review_workflow = self._build_review_workflow()
        # 合同生成工作流
        self.generation_workflow = self._build_generation_workflow()
    
    def _build_review_workflow(self):
        """构建审查交互工作流"""
        workflow: StateGraph[ContractState] = StateGraph(ContractState)
        
        # 添加节点
        workflow.add_node("initial_analysis", self._initial_analysis_node)
        workflow.add_node("user_input_round0", self._user_input_round0_node)
        workflow.add_node("continue_analysis", self._continue_analysis_node)
        workflow.add_node("check_completeness", self._check_completeness_node)
        workflow.add_node("user_input_roundN", self._user_input_roundN_node)
        
        # 设置入口
        workflow.set_entry_point("initial_analysis")
        
        # 添加边
        workflow.add_edge("initial_analysis", "user_input_round0")
        workflow.add_edge("user_input_round0", "continue_analysis")
        workflow.add_conditional_edges(
            "continue_analysis",
            self._should_continue_review,
            {
                "continue": "check_completeness",
                "complete": END
            }
        )
        workflow.add_conditional_edges(
            "check_completeness",
            self._should_ask_user,
            {
                "ask": "user_input_roundN",
                "complete": END
            }
        )
        workflow.add_edge("user_input_roundN", "continue_analysis")
        
        return workflow.compile()
    
    def _build_generation_workflow(self):
        """构建合同生成工作流"""
        workflow: StateGraph[ContractState] = StateGraph(ContractState)
        
        workflow.add_node("prepare_generation", self._prepare_generation_node)
        workflow.add_node("generate_contract", self._generate_contract_node)
        workflow.add_node("check_length", self._check_length_node)
        workflow.add_node("split_generate", self._split_generate_node)
        workflow.add_node("merge_contract", self._merge_contract_node)
        
        workflow.set_entry_point("prepare_generation")
        
        workflow.add_edge("prepare_generation", "generate_contract")
        workflow.add_conditional_edges(
            "check_length",
            self._should_split,
            {
                "split": "split_generate",
                "merge": "merge_contract"
            }
        )
        workflow.add_edge("generate_contract", "check_length")
        workflow.add_edge("split_generate", "merge_contract")
        workflow.add_edge("merge_contract", END)
        
        return workflow.compile()
    
    # 审查工作流节点
    async def _initial_analysis_node(self, state: ContractState):
        """初始分析节点"""
        prompt = self._prepare_initial_context(state["input_data"], "", state["asked_questions"], is_first_round=True)
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        ai_response = response.content.strip()[:300]
        
        new_questions = self._extract_questions(ai_response)
        state["asked_questions"].extend(new_questions)
        state["conversation_history"].append({
            "role": "assistant",
            "content": ai_response,
            "name": "SemanticReviewer"
        })
        state["info_complete"] = False  # 第0轮强制不完整
        
        print(f"\n{'=' * 60}")
        print("🤖 AI分析结果：")
        print(ai_response)
        print(f"{'=' * 60}")
        
        return state
    
    async def _user_input_round0_node(self, state: ContractState):
        """第0轮用户输入节点"""
        print("\n💬 请根据以上问题补充信息（多行输入，空行结束，输入exit可终止）：")
        user_input = self._get_user_input().strip()
        
        if not user_input:
            logger.warning("⚠️ 用户未提供补充信息")
        elif user_input.lower() in {"exit", "quit", "q", "停止", "退出"}:
            raise ValueError("用户终止对话")
        else:
            state["user_notes"].append(user_input)
            state["conversation_history"].append({
                "role": "user",
                "content": user_input,
                "name": "User"
            })
            state["round_idx"] += 1
        
        return state
    
    async def _user_input_roundN_node(self, state: ContractState):
        """第N轮用户输入节点"""
        print("\n💬 请根据以上问题补充信息（多行输入，空行结束，输入exit可终止）：")
        user_input = self._get_user_input().strip()
        
        if not user_input:
            logger.warning("⚠️ 用户未提供补充信息")
        elif user_input.lower() in {"exit", "quit", "q", "停止", "退出"}:
            raise ValueError("用户终止对话")
        else:
            state["user_notes"].append(user_input)
            state["conversation_history"].append({
                "role": "user",
                "content": user_input,
                "name": "User"
            })
            state["round_idx"] += 1
        
        return state
    
    async def _continue_analysis_node(self, state: ContractState):
        """继续分析节点"""
        if state["round_idx"] >= state["max_rounds"]:
            state["info_complete"] = False  # 达到最大轮次，标记为不完整但继续生成
            return state
        
        print(f"\n🔄 第 {state['round_idx']} 轮：基于用户回答继续分析")
        merged_supplements = "\n".join(f"- {note}" for note in state["user_notes"])
        prompt = self._prepare_initial_context(
            state["input_data"], 
            merged_supplements, 
            state["asked_questions"], 
            is_first_round=False
        )
        
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        ai_response = response.content.strip()[:300]
        state["info_complete"] = self._should_stop_questioning(ai_response)
        
        new_questions = self._extract_questions(ai_response)
        state["asked_questions"].extend(new_questions)
        state["conversation_history"].append({
            "role": "assistant",
            "content": ai_response,
            "name": "SemanticReviewer"
        })
        
        return state
    
    async def _check_completeness_node(self, state: ContractState):
        """检查完整性节点"""
        # 只有不完整时才输出到终端
        if not state["info_complete"]:
            print(f"\n{'=' * 60}")
            print("🤖 AI分析结果：")
            print(state["conversation_history"][-1]["content"])
            print(f"{'=' * 60}")
            
            if state["round_idx"] >= state["max_rounds"]:
                print("\n⚠️ 已达到最大对话轮次，将基于现有信息生成合同范本")
        else:
            print(f"\n{'=' * 60}")
            print("✅ 信息完整，开始生成合同")
            print("[DEBUG] 🤖 AI分析结果：")
            print(state["conversation_history"][-1]["content"])
            print(f"{'=' * 60}")
        
        return state
    
    def _should_continue_review(self, state: ContractState) -> str:
        """判断是否继续审查"""
        if state["info_complete"] or state["round_idx"] >= state["max_rounds"]:
            return "complete"
        return "continue"
    
    def _should_ask_user(self, state: ContractState) -> str:
        """判断是否需要询问用户"""
        # 如果不完整且未达到最大轮次，需要询问用户
        if not state["info_complete"] and state["round_idx"] < state["max_rounds"]:
            return "ask"
        return "complete"
    
    # 生成工作流节点
    async def _prepare_generation_node(self, state: ContractState):
        """准备生成节点"""
        final_data = state["input_data"].copy()
        if state["user_notes"]:
            final_data["user_supplements"] = "\n\n".join(state["user_notes"])
        
        # 支持外部指定模板
        template_key = state.get("template_key")
        contract_type = state["input_data"].get('contract_type', '').lower()
        selected_template = self._select_template(contract_type, template_key)
        state["template_key"] = selected_template
        
        return state
    
    async def _generate_contract_node(self, state: ContractState):
        """生成合同节点"""
        generation_prompt = self._build_generation_prompt(state["input_data"], state["template_key"])
        response = await self.llm.ainvoke([HumanMessage(content=generation_prompt)])
        state["contract_content"] = response.content.strip()
        
        return state
    
    async def _check_length_node(self, state: ContractState):
        """检查长度节点"""
        # 长度检查逻辑在条件边中处理
        return state
    
    def _should_split(self, state: ContractState) -> str:
        """判断是否需要分段"""
        if self._check_contract_length(state["contract_content"]):
            return "split"
        return "merge"
    
    async def _split_generate_node(self, state: ContractState):
        """分段生成节点"""
        generation_prompt = self._build_generation_prompt(state["input_data"], state["template_key"])
        segments = self._split_contract_generation(generation_prompt)
        
        contract_segments = []
        for segment_prompt in segments:
            segment_content = await self._generate_contract_segment(segment_prompt)
            contract_segments.append(segment_content)
        
        state["contract_content"] = self._merge_contract_segments(contract_segments)
        return state
    
    async def _merge_contract_node(self, state: ContractState):
        """合并合同节点"""
        if not self._check_contract_length(state["contract_content"]):
            # 如果不需要分段，添加声明即可
            if not state["contract_content"].startswith("---"):
                header = """---
【重要提示】本合同由AI自动生成，仅为参考范本，必须经过专业法律人员审核和修改后方可使用。不得直接使用，否则可能产生法律风险。
---

"""
                state["contract_content"] = header + state["contract_content"]
        
        return state
    
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

    def _build_generation_prompt(self, input_data: Dict[str, Any], template_key: Optional[str] = None) -> str:
        """构建合同生成提示词"""
        if template_key is None:
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

**重要**：参考模板的结构和条款类型，并填空修改，但不要直接完全复制模板内容。基于提供的具体信息生成全新的合同内容。"""

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
    
    def _get_user_input(self) -> str:
        """获取用户多行输入（空行结束）"""
        lines = []
        print("请输入内容（空行结束）：")
        while True:
            try:
                line = input()
                if not line.strip():
                    break
                lines.append(line)
            except EOFError:
                break
        return "\n".join(lines)
    
    async def process_contract_interactive(self, input_data: Dict[str, Any], template_key: Optional[str] = None) -> Dict[str, Any]:
        """交互式处理合同请求（使用LangGraph工作流）"""
        initial_state: ContractState = {
            "input_data": input_data,
            "conversation_history": [],
            "user_notes": [],
            "asked_questions": [],
            "round_idx": 0,
            "info_complete": False,
            "contract_content": "",
            "template_key": template_key,
            "max_rounds": self.max_rounds
        }
        
        try:
            # 执行审查工作流
            print(f"\n{'=' * 60}")
            print("🤖 AI正在执行语义级完整性分析...")
            print(f"{'=' * 60}")
            print(f"\n🔄 第 0 轮：初始分析")
            
            review_state = await self.review_workflow.ainvoke(initial_state)
            
            # 执行生成工作流
            if review_state.get("info_complete"):
                print("\n🎯 信息完整，开始生成合同范本...")
            else:
                print("\n🎯 基于现有信息生成合同范本（请注意：此范本需要进一步修改完善）...")
            
            final_state = await self.generation_workflow.ainvoke(review_state)
            
            return {
                "status": "success",
                "contract": final_state["contract_content"],
                "conversation": final_state["conversation_history"],
                "message": "合同生成成功"
            }
            
        except ValueError as e:
            if "用户终止对话" in str(e):
                return {
                    "status": "error",
                    "message": "用户终止对话",
                    "conversation": initial_state.get("conversation_history", [])
                }
            raise
        except Exception as e:
            logger.error(f"❌ 合同处理失败: {e}")
            raise

    def run(self, template_key: Optional[str] = None):
        """运行演示主函数（同步入口，内部调用异步逻辑）"""
        asyncio.run(self._async_run(template_key))

    async def _async_run(self, template_key: Optional[str] = None):
        """异步核心主函数（交互式处理模式）"""
        try:
            logger.info("🚀 启动合同生成演示程序（LangChain版）")
            print("=" * 70)
            print("🤖 智能合同生成系统（LangChain + LangGraph）")
            print("=" * 70)

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

            # 新增：显示模板信息
            if template_key:
                if template_key in self.templates:
                    print(f"✅ 使用指定模板: {template_key}")
                else:
                    print(f"⚠️ 指定模板 '{template_key}' 不存在，将使用自动选择")
                    template_key = None
            else:
                print("✅ 将使用自动模板选择")

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

                # 修改：传递指定的模板
                result = await self.process_contract_interactive(input_data, template_key)

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
        if self.memory is not None:
            self.memory.clear()
    
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
    template_name = "物联网流量合同"  # 在这里设置想要的模板名称
    manager = LangChainContractManager()
    manager.run(template_key=template_name)
