import json
import yaml

from dataclasses import dataclass
from huggingface_hub import hf_hub_download
from typing import Optional


DEFAULT_PARAM_CONFIG_PATH = "./param_config.yaml"
DEFAULT_MODEL_CONFIG_PATH = "./model_config.yaml"

MODEL_NAME_TO_HF = {
    "deepseek/deepseek-r1-0528": "deepseek-ai/DeepSeek-R1-0528",
    "deepseek/deepseek-v3.1": "deepseek-ai/DeepSeek-V3.1",
    "deepseek/deepseek-v3.2": "deepseek-ai/DeepSeek-V3.2",
    "minimax/minimax-m2": "MiniMaxAI/MiniMax-M2",
    "minimax/minimax-m2.1": "MiniMaxAI/MiniMax-M2.1",
    "minimax/minimax-m2.5": "MiniMaxAI/MiniMax-M2.5",
    "moonshotai/kimi-k2-0905": "moonshotai/Kimi-K2-Instruct-0905",
    "moonshotai/kimi-k2-thinking": "moonshotai/Kimi-K2-Thinking",
    "moonshotai/kimi-k2.5": "moonshotai/Kimi-K2.5",
    "openai/gpt-oss-120b": "openai/gpt-oss-120b",
    "qwen/qwen3-coder-480b-a35b-instruct": "Qwen/Qwen3-Coder-480B-A35B-Instruct",
    "qwen/qwen3-vl-235b-a22b-instruct": "Qwen/Qwen3-VL-235B-A22B-Instruct",
    "qwen/qwen3.5-397b-a17b": "Qwen/Qwen3.5-397B-A17B",
    "zai-org/glm-4.5-air": "zai-org/GLM-4.5-Air",
    "zai-org/glm-4.6": "zai-org/GLM-4.6",
    "zai-org/glm-4.7": "zai-org/GLM-4.6",
    "zai-org/glm-5": "zai-org/GLM-5",
    "zai-org/glm-5-turbo": "zai-org/GLM-5-FP8",
}


@dataclass
class ModelMetrics:
    model: str = None
    param_size: int = None
    ttft: float = None
    tpot: float = None
    tps: float = None
    toolcall_error_rate: float = None  # percentage
    structured_output_error_rate: float = None  # percentage


@dataclass
class ModelInfo:
    """
    Abstraction of transformer model architecture for scheduling decisions.
    Assuming all decoder layers have uniform computational and memory requirements
    """

    head_size: int
    hidden_dim: int
    intermediate_dim: int
    num_attention_heads: int
    num_kv_heads: int
    vocab_size: int
    num_layers: int
    ffn_num_projections: int = 3
    num_local_experts: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    moe_intermediate_dim: Optional[int] = None
    tie_embedding: bool = False
    # Default int8
    param_bytes_per_element: float = 1
    cache_bytes_per_element: int = 1
    embedding_bytes_per_element: int = 1

    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    head_size_k: int = None
    head_size_v: int = None

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.qk_nope_head_dim is not None and self.qk_rope_head_dim is not None:
            self.head_size_k = self.qk_nope_head_dim + self.qk_rope_head_dim
        else:
            self.head_size_k = self.head_size
        self.head_size_v = self.head_size

    @property
    def q_dim(self) -> int:
        """Return query head dim."""
        return self.num_attention_heads * self.head_size

    @property
    def v_dim(self) -> int:
        """Return key and value head dim."""
        return self.num_kv_heads * self.head_size_v

    @property
    def k_dim(self) -> int:
        """Return key head dim."""
        return self.num_kv_heads * self.head_size_k

    @property
    def embedding_io_bytes(self) -> int:
        """Estimate memory for input_embeddings / or lm_head."""
        return self.embedding_bytes_per_element * self.vocab_size * self.hidden_dim

    @property
    def per_token_per_layer_kv_size(self) -> int:
        """Return bytes per token for KV cache."""
        return self.cache_bytes_per_element * (self.k_dim + self.v_dim)

    def per_layer_kv_cache_size(self, *, batch_size: int = 1, source_seq_len: int = 256) -> int:
        """Return size of KV cache in bytes for given request dimensions."""
        return self.per_token_per_layer_kv_size * batch_size * source_seq_len

    def expected_num_activated_experts(
        self, *, batch_size: int = 1, target_seq_len: int = 1
    ) -> Optional[int]:
        """Return expected number of activated experts for a request size."""
        num_tokens = batch_size * target_seq_len
        if self.num_local_experts is not None and self.num_experts_per_tok is not None:
            return int(
                self.num_local_experts
                * (1 - (1 - self.num_experts_per_tok / self.num_local_experts) ** num_tokens)
            )
        return None

    def decoder_layer_io_bytes(
        self,
        roofline: Optional[bool] = None,
        *,
        batch_size: int = 1,
        target_seq_len: int = 1,
        source_seq_len: int = 256,
    ) -> int:
        """
        Estimate memory per decoder layer in bytes including params and kv cache.

        Args:
            roofline: True if calculation is for roofline io latency, otherwise for param size estimation.
            batch_size: Request batch size
            target_seq_len: Target sequence length (tokens to generate)
            source_seq_len: Source sequence length (prompt tokens)
        """
        # Attention params
        qo_params = self.param_bytes_per_element * self.hidden_dim * self.q_dim * 2
        kv_params = self.param_bytes_per_element * self.hidden_dim * (self.k_dim + self.v_dim)
        attention_params = qo_params + kv_params

        # FFN params
        ffn_params = self.param_bytes_per_element * self.ffn_num_projections * self.hidden_dim
        if self.moe_intermediate_dim is not None:
            ffn_params *= self.moe_intermediate_dim
        else:
            ffn_params *= self.intermediate_dim

        if roofline:
            expected_experts = self.expected_num_activated_experts(
                batch_size=batch_size, target_seq_len=target_seq_len
            )
            if expected_experts is not None:
                ffn_params *= expected_experts
            kv_cache_size = self.per_layer_kv_cache_size(
                batch_size=batch_size, source_seq_len=source_seq_len
            )
        else:
            if self.num_local_experts is not None:
                ffn_params *= self.num_local_experts
            kv_cache_size = 0

        return round(ffn_params + kv_cache_size + attention_params)

    def lm_head_flops(self, target_seq_len: int = 1) -> int:
        """Estimate FLOPs for lm_head (last layer GEMM) for a sequence length."""
        return 2 * target_seq_len * self.hidden_dim * self.vocab_size


def get_model_info(config):
    # text config
    if config.get("hidden_size", None) is None:
        config = config.get("text_config", {})

    quant_method = config.get("quant_method", None)
    quantization_config = config.get("quantization_config", None)
    if quant_method is None and quantization_config is not None:
        quant_method = quantization_config.get("quant_method", None)

    if quant_method is None:
        param_bytes_per_element = 2
    elif quant_method in ("fp8", "compressed-tensors"):
        param_bytes_per_element = 1
    elif quant_method in ("mxfp4", "int4", "awq", "gptq"):
        param_bytes_per_element = 0.5
    else:
        param_bytes_per_element = 1

    # get local experts
    num_local_experts = config.get("num_local_experts", None)
    if num_local_experts is None:
        num_local_experts = config.get("num_experts", None)
    if num_local_experts is None:
        num_local_experts = config.get("n_routed_experts", None)

    model_info = ModelInfo(
        head_size=config.get("head_dim", 128),
        qk_nope_head_dim=config.get("qk_nope_head_dim", None),
        qk_rope_head_dim=config.get("qk_rope_head_dim", None),
        hidden_dim=config.get("hidden_size", 0),
        intermediate_dim=config.get("intermediate_size", 0),
        num_attention_heads=config.get("num_attention_heads", 0),
        num_kv_heads=config.get("num_key_value_heads", 0),
        vocab_size=config.get("vocab_size", 0),
        num_layers=config.get("num_hidden_layers", 0),
        ffn_num_projections=3,
        param_bytes_per_element=param_bytes_per_element,
        cache_bytes_per_element=2,
        embedding_bytes_per_element=2,
        num_local_experts=num_local_experts,
        num_experts_per_tok=config.get("num_experts_per_tok", None),
        moe_intermediate_dim=config.get("moe_intermediate_size", None),
    )
    return model_info


def calculate_model_param_size(config):
    model_info = get_model_info(config)
    param_size = (
            model_info.embedding_io_bytes
            + model_info.num_layers * model_info.decoder_layer_io_bytes(roofline=False)
        ) * 1.0 / 1024 / 1024 / 1024
    return param_size


def print_model_metrics(model_metrics):
    print("---- Model Metrics ----")
    print(f"model: {model_metrics.model}")
    print(f"param_size: {model_metrics.param_size}")
    print(f"ttft: {model_metrics.ttft}s")
    print(f"tpot: {model_metrics.tpot}s")
    print(f"tps: {model_metrics.tps}/batch")
    print(f"toolcall_error_rate: {model_metrics.toolcall_error_rate}%")
    print(f"structured_output_error_rate: {model_metrics.structured_output_error_rate}%")


def load_default_param_config():
    default_metrics = []
    with open(DEFAULT_PARAM_CONFIG_PATH) as file:
        config_table = yaml.safe_load(file)
        for table in config_table["metrics"]:
            metrics = ModelMetrics(
                param_size = table["param_size"],
                ttft=table["ttft"],
                tpot=table["tpot"],
                tps=table["tps"],
                toolcall_error_rate=table["toolcall_error_rate"],
                structured_output_error_rate=table["structured_output_error_rate"],
            )
            default_metrics.append(metrics)
    return default_metrics


def load_default_model_config():
    model_metrics = {}
    with open(DEFAULT_MODEL_CONFIG_PATH) as file:
        config_table = yaml.safe_load(file)
        for model_name in config_table:
            model = config_table[model_name]
            metrics = ModelMetrics(
                model=model_name,
                param_size=0,
                ttft=model["ttft"],
                tpot=model["tpot"],
                tps=model["tps"],
                toolcall_error_rate=model["toolcall_error_rate"],
                structured_output_error_rate=model["structured_output_error_rate"],
            )
            model_metrics[model_name] = metrics
    return model_metrics


def get_model_config(model_name):
    config = None
    hf_model_name = MODEL_NAME_TO_HF.get(model_name, None)
    if hf_model_name is None:
        hf_model_name = model_name
    try:
        config_file = hf_hub_download(repo_id=hf_model_name, filename="config.json")
        with open(config_file, 'r') as f:
            config = json.load(f)
    except Exception as e:
        pass

    return config


def get_model_metrics(model_name):
    """Get model performance metrics from model name"""
    default_param_metrics = load_default_param_config()
    default_model_metrics = load_default_model_config()

    model_config = get_model_config(model_name)
    metrics = None
    param_size = 0
    if model_config is None:
        if model_name in default_model_metrics:
            metrics = default_model_metrics[model_name]
        else:
            metrics = default_param_metrics[-1]
    else:
        param_size = calculate_model_param_size(model_config)
        for ref_metrics in default_param_metrics:
            if param_size < ref_metrics.param_size:
                metrics = ref_metrics
                break
        if metrics is None:
            metrics = default_param_metrics[-1]

    model_metrics = ModelMetrics(
        model=model_name,
        param_size=param_size,
        ttft=metrics.ttft,
        tpot=metrics.tpot,
        tps=metrics.tps,
        toolcall_error_rate=metrics.toolcall_error_rate,
        structured_output_error_rate=metrics.structured_output_error_rate,
    )

    return model_metrics


def main():
    model_names = [
        "anthropic/claude-haiku-4-5",
        "anthropic/claude-opus-4-5",
        "anthropic/claude-opus-4-5-eco",
        "anthropic/claude-opus-4-6",
        "anthropic/claude-sonnet-4-5",
        "anthropic/claude-sonnet-4-5-eco",
        "anthropic/claude-sonnet-4-6",
        "deepseek/deepseek-r1-0528",
        "deepseek/deepseek-v3.1",
        "deepseek/deepseek-v3.2",
        "google/gemini-2.5-flash",
        "google/gemini-2.5-flash-image",
        "google/gemini-2.5-pro",
        "google/gemini-3-flash-preview",
        "google/gemini-3-pro-image-preview",
        "google/gemini-3-pro-preview",
        "google/gemini-3.1-flash-image-preview",
        "google/gemini-3.1-flash-lite-preview",
        "google/gemini-3.1-pro-preview",
        "minimax/minimax-m2",
        "minimax/minimax-m2.1",
        "minimax/minimax-m2.5",
        "minimax/minimax-m2.7",
        "moonshotai/kimi-k2-0905",
        "moonshotai/kimi-k2-thinking",
        "moonshotai/kimi-k2.5",
        "openai/gpt-4.1",
        "openai/gpt-4o-mini",
        "openai/gpt-5",
        "openai/gpt-5.2",
        "openai/gpt-5.3-codex",
        "openai/gpt-5.4-2026-03-05",
        "openai/gpt-5.4-mini-2026-03-17",
        "openai/gpt-5.4-nano-2026-03-17",
        "openai/gpt-5.4-pro-2026-03-05",
        "openai/gpt-oss-120b",
        "qwen/qwen3-coder-480b-a35b-instruct",
        "qwen/qwen3-vl-235b-a22b-instruct",
        "qwen/qwen3.5-397b-a17b",
        "x-ai/grok-4-1-fast-non-reasoning",
        "x-ai/grok-4.1-fast-reasoning",
        "x-ai/grok-code-fast-1",
        "xiaomi/mimo-v2-omni",
        "xiaomi/mimo-v2-pro",
        "zai-org/glm-4.5-air",
        "zai-org/glm-4.6",
        "zai-org/glm-4.7",
        "zai-org/glm-5",
        "zai-org/glm-5-turbo",
    ]

    for model_name in model_names:
        model_metrics = get_model_metrics(model_name)
        print_model_metrics(model_metrics)


if __name__ == "__main__":
    main()
