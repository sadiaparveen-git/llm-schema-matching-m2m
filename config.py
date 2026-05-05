import os
from dotenv import load_dotenv

load_dotenv()

config = {
    "OPENAI_API_KEY":    os.environ.get("OPENAI_API_KEY", None),
    "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", None),
    "LLM_PROVIDER":      os.environ.get("LLM_PROVIDER", "openai"),
    "LLM_MODEL":         os.environ.get("LLM_MODEL", None),
    "ANTHROPIC_MODEL":   os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
    "OPENAI_MODEL":      os.environ.get("OPENAI_MODEL", "gpt-4-turbo"),
    "MAX_GROUP_SIZE":    int(os.environ.get("MAX_GROUP_SIZE", "2")),
    "PHASE_2_ENABLED":   os.environ.get("PHASE_2_ENABLED", "True").lower() == "true",
    "PHASE_1_CONFIDENCE": float(os.environ.get("PHASE_1_CONFIDENCE", "0.66")),
    "MATCH_SCORE_THRESHOLD": float(os.environ.get("MATCH_SCORE_THRESHOLD", "0.5")),
    "QUERY_LLM":         os.environ.get("QUERY_LLM", "True").lower() == "true",
    "OPENAI_N":          int(os.environ.get("OPENAI_N", "3")),
    "OPENAI_TEMPERATURE": float(os.environ.get("OPENAI_TEMPERATURE", "1.0")),
    "RESULTS_DIR":       os.environ.get("RESULTS_DIR", "thesis-extension/results"),
    "TEMPLATE_DIR":      os.environ.get("TEMPLATE_DIR", "thesis-extension/templates"),
    "PARALLEL_OPENAI_REQUESTS":    int(os.environ.get("PARALLEL_OPENAI_REQUESTS", "5")),
    "PARALLEL_ANTHROPIC_REQUESTS": int(os.environ.get("PARALLEL_ANTHROPIC_REQUESTS", "3")),
}


class ConfigurationError(RuntimeError):
    pass


def get_provider():
    from llm_provider import OpenAIProvider, AnthropicProvider  # imported lazily to avoid circular deps

    name = config["LLM_PROVIDER"]
    if name == "openai":
        if not config["OPENAI_API_KEY"]:
            raise ConfigurationError(
                "LLM_PROVIDER=openai but OPENAI_API_KEY is not set. "
                "Add it to thesis-extension/.env or export it in the shell."
            )
        return OpenAIProvider()
    if name == "anthropic":
        if not config["ANTHROPIC_API_KEY"]:
            raise ConfigurationError(
                "LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set. "
                "Add it to thesis-extension/.env or export it in the shell."
            )
        return AnthropicProvider()
    raise ConfigurationError(f"Unknown LLM_PROVIDER: {name!r}")
