import os
from loguru import logger
import torch  # 提前导入torch，避免加载时可能的导入顺序问题

from singleton import singleton


@singleton
class QwenModel:  # 类名更新为QwenModel，明确模型类型
    """
    使用单例模式管理Qwen模型（Qwen2.5-7B）
    采用8bit量化（兼顾质量和显存，16GB显存适配）
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._is_loaded = False  # 移除processor相关属性

    def load_model(self):
        """加载Qwen模型（8bit量化 + 适配最新transformers API）"""
        if self._is_loaded:
            logger.info("📦 模型已加载，跳过重复加载")
            return True

        try:
            # 导入必要的组件，Qwen是因果语言模型，使用AutoModelForCausalLM
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            # 模型本地路径（确保指向Qwen模型的实际目录）
            MODEL_ROOT = "/eos_pool/build/modelscope/models/Qwen3-14B"

            # 检查模型路径是否存在
            if not os.path.exists(MODEL_ROOT):
                logger.error(f"❌ Qwen模型路径不存在: {MODEL_ROOT}")
                return False

            logger.info(f"⏳ 正在从 {MODEL_ROOT} 加载模型{MODEL_ROOT.split('/')[-1]}...")

            # 1. 加载tokenizer（Qwen需要trust_remote_code=True）
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_ROOT,
                trust_remote_code=True,
                padding_side="left"  # Qwen推荐left padding，避免生成截断
            )
            # 补充pad_token（Qwen部分版本默认未定义，用eos_token替代）
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # 2. 配置8bit量化参数（替换原4bit配置，适配BitsAndBytes）
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,  # 核心：启用8bit量化（替换load_in_4bit）
                bnb_8bit_use_double_quant=True,  # 双量化优化，兼顾显存和质量
                bnb_8bit_quant_type="nf8",  # 8bit推荐量化类型（nf8适配语义任务）
                bnb_8bit_compute_dtype=torch.float16,  # 计算精度，避免质量损失
                bnb_8bit_quant_storage=torch.float16,  # 量化存储精度，节省显存
                bnb_8bit_use_llm_int8_skip_modules=["lm_head"]  # 跳过输出层量化，提升生成质量
            )

            bnb_config_4 = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_storage = torch.float16,  # 量化存储精度，节省显存
                bnb_4bit_use_llm_int8_skip_modules = ["lm_head"]  # 跳过输出层量化，提升生成质量
            )

            # 3. 加载模型（使用AutoModelForCausalLM，适配语言模型）
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_ROOT,
                dtype=torch.float16,  # 基础计算精度
                trust_remote_code=True,  # Qwen需要加载自定义代码
                device_map="auto",  # 自动分配设备（优先GPU）
                quantization_config=bnb_config_4,  # 传入bit量化配置8/
                low_cpu_mem_usage=True,  # 减少CPU内存占用
                offload_buffers=False,  # 新增：卸载非核心缓冲区到CPU，进一步节省GPU显存
                offload_state_dict=False
            )

            self._is_loaded = True
            logger.info(f"✅ {MODEL_ROOT.split('/')[-1]}模型加载成功")
            return True

        except Exception as e:
            logger.error(f"❌模型加载失败: {e}", exc_info=True)
            # 加载失败时重置状态
            self.model = None
            self.tokenizer = None
            self._is_loaded = False
            return False

    def release_resources(self):
        """释放模型资源"""
        if not self._is_loaded:
            logger.info("📭 模型未加载，无需释放")
            return

        try:
            import gc

            logger.info("🔄 开始释放模型资源...")

            # 清除模型和tokenizer
            if self.model is not None:
                del self.model
                self.model = None
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None

            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("🧹 已清理GPU缓存")

            # 强制垃圾回收
            gc.collect()

            self._is_loaded = False
            logger.info("✅ 模型资源释放完成")

        except Exception as e:
            logger.error(f"❌ 资源释放失败: {e}")

    def is_loaded(self):
        """检查模型是否已加载"""
        return self._is_loaded

    def get_model(self):
        """获取模型组件（仅返回model和tokenizer）"""
        if not self._is_loaded:
            logger.warning("⚠️ 模型未加载，请先调用load_model()")
            return None, None
        return self.model, self.tokenizer

    def get_model_info(self):
        """获取模型信息（补充量化类型）"""
        return {
            'model_loaded': self.model is not None,
            'tokenizer_loaded': self.tokenizer is not None,
            'is_loaded': self._is_loaded,
            'quantization': '8bit'  # 明确标注量化类型
        }