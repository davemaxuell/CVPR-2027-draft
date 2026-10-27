from attr import dataclass
from agentkit.llm import LLMCallable, Message
from agentkit.utils import load_config

# Load default values from config
CONFIG: dict = load_config("configs/configs.yaml")
LLM_CONFIG: dict = CONFIG.get("llm_config", {})
DEFAULT_MAX_TOKEN = LLM_CONFIG.get("max_token", 512)
DEFAULT_TEMPERATURE = LLM_CONFIG.get("temperature", 0.1)
DEFAULT_NUM_COMPS = LLM_CONFIG.get("num_comps", 1)

@dataclass
class ReasoningConfig:
    temperature: float = None
    max_tokens: int = None
    stop_strs: list[str] = None
    num_comps: int = None

class ReasoningBase:
    def __init__(self, llm_model: LLMCallable):
        self.llm_model: LLMCallable = llm_model
    
    def __call__(self, prompts: list[Message], config: ReasoningConfig) -> str:
        raise NotImplementedError("This method should be implemented by subclasses.")



class ReasoningIO(ReasoningBase):
    """
    A subclass of ReasoningBase that defines how to perform a reasoning call
    just by using an underlying language model (LLM).
    """

    def __call__(self, prompts: list[Message], config: ReasoningConfig) -> str:

        # Use default values if config parameters are None
        reasoning_result = self.llm_model(
            prompts,
            temperature=config.temperature if config.temperature is not None else DEFAULT_TEMPERATURE,
            max_tokens=config.max_tokens if config.max_tokens is not None else DEFAULT_MAX_TOKEN,
            stop_strs=config.stop_strs,
            num_comps=config.num_comps if config.num_comps is not None else DEFAULT_NUM_COMPS
        )
        return reasoning_result