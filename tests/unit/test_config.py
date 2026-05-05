import os
from unittest.mock import patch

from cbserver.config import Settings, get_settings


def test_defaults_match_plan() -> None:
    settings = Settings()
    assert settings.model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert settings.block_size == 16
    assert settings.max_num_batched_tokens == 2048


def test_env_overrides_apply() -> None:
    with patch.dict(os.environ, {"CBSERVER_MODEL": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "CBSERVER_PORT": "9000"}):
        settings = get_settings()
        assert settings.model == "HuggingFaceTB/SmolLM2-1.7B-Instruct"
        assert settings.port == 9000
