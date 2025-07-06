from unittest.mock import MagicMock, patch

import pytest

from memos.configs.mem_os import MOSConfig
from memos.mem_os.main import MOS


@pytest.fixture
def simple_config():
    """Simple configuration for testing"""
    return MOSConfig(
        user_id="test_user",
        session_id="test_session",
        chat_model={
            "backend": "huggingface",
            "config": {
                "model_name_or_path": "test-model",
                "temperature": 0.1,
                "max_tokens": 100,
            },
        },
        mem_reader={
            "backend": "simple_struct",
            "config": {
                "llm": {
                    "backend": "ollama",
                    "config": {
                        "model_name_or_path": "test-model",
                        "temperature": 0.8,
                        "max_tokens": 100,
                    },
                },
                "embedder": {
                    "backend": "ollama",
                    "config": {
                        "model_name_or_path": "test-embed",
                    },
                },
                "chunker": {
                    "backend": "sentence",
                    "config": {
                        "tokenizer_or_token_counter": "gpt2",
                        "chunk_size": 512,
                        "chunk_overlap": 128,
                        "min_sentences_per_chunk": 1,
                    },
                },
            },
        },
        enable_textual_memory=True,
        enable_activation_memory=False,
        enable_parametric_memory=False,
        top_k=5,
        max_turns_window=10,
    )


@patch("memos.mem_os.core.UserManager")
@patch("memos.mem_os.core.MemReaderFactory")
@patch("memos.mem_os.core.LLMFactory")
def test_mos_can_initialize(mock_llm, mock_reader, mock_user_manager, simple_config):
    """Test that MOS can be initialized successfully"""
    # Mock all dependencies
    mock_llm.from_config.return_value = MagicMock()
    mock_reader.from_config.return_value = MagicMock()

    user_manager_instance = MagicMock()
    user_manager_instance.validate_user.return_value = True
    mock_user_manager.return_value = user_manager_instance

    # Create MOS instance
    mos = MOS(simple_config)

    # Basic assertions
    assert mos is not None
    assert mos.user_id == "test_user"


@patch("memos.mem_os.core.UserManager")
@patch("memos.mem_os.core.MemReaderFactory")
@patch("memos.mem_os.core.LLMFactory")
def test_mos_has_core_methods(mock_llm, mock_reader, mock_user_manager, simple_config):
    """Test that MOS inherits methods from MOSCore"""
    # Mock all dependencies
    mock_llm.from_config.return_value = MagicMock()
    mock_reader.from_config.return_value = MagicMock()

    user_manager_instance = MagicMock()
    user_manager_instance.validate_user.return_value = True
    mock_user_manager.return_value = user_manager_instance

    # Create MOS instance
    mos = MOS(simple_config)

    # Check that key methods exist and are callable
    assert hasattr(mos, "chat")
    assert hasattr(mos, "search")
    assert hasattr(mos, "add")
    assert callable(mos.chat)
    assert callable(mos.search)
    assert callable(mos.add)
