"""Smoke test UI Streamlit dengan AppTest (tanpa browser, tanpa jaringan)."""

from __future__ import annotations

from pathlib import Path

import pytest

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import app  # noqa: E402
import pipeline  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app.py"


def _fresh_app() -> AppTest:
    return AppTest.from_file(str(APP), default_timeout=60)


def test_app_boots_without_exception():
    at = _fresh_app().run()
    assert not at.exception
    assert any("SONYBAE CLIPS" in str(item.value) for item in at.markdown)
    assert "🚀 Proses Klip Viral SONYBAE!" in [button.label for button in at.button]
    assert "🧹 Hapus API key & cookies dari sesi" in [button.label for button in at.button]


def test_app_has_configuration_widgets():
    at = _fresh_app().run()
    labels = [widget.label for widget in at.sidebar.text_input]
    assert "API Key" in labels
    assert "Base URL" in labels
    assert "Model" in labels
    assert at.radio[0].value == "AI pilih momen"


def test_app_blocks_empty_url():
    at = _fresh_app().run()
    at.button[0].click().run()
    assert not at.exception
    assert any("kosong" in str(err.value).lower() for err in at.error)


def test_app_rejects_non_youtube_url():
    at = _fresh_app().run()
    at.text_input[0].set_value("https://evil.example.com/watch?v=1")
    at.button[0].click().run()
    assert not at.exception
    assert any("youtube" in str(err.value).lower() for err in at.error)


def test_humanize_transcript_error_does_not_mislabel_http_429():
    error = pipeline.TranscriptError(
        "YouTube membatasi permintaan subtitle (HTTP 429); gunakan Timestamp manual."
    )
    message = app._humanize_error(error)
    assert "HTTP 429" in message
    assert "API key AI" not in message
    assert "pilih video yang punya subtitle" not in message


def test_app_requires_api_key_in_ai_mode():
    at = _fresh_app().run()
    at.text_input[0].set_value("https://youtu.be/abc123")
    primary = next(button for button in at.button if button.label.startswith("🚀"))
    primary.click().run()
    assert not at.exception
    assert any("api key" in str(err.value).lower() for err in at.error)


def test_app_blocks_user_custom_private_ai_endpoint():
    at = _fresh_app().run()
    at.text_input[0].set_value("https://youtu.be/abc123")
    at.selectbox[0].select("Custom / lainnya").run()
    at.secrets["OPENAI_API_KEY"] = "server-owned-secret"
    api_key = next(widget for widget in at.sidebar.text_input if widget.label == "API Key")
    api_key.set_value("user-own-key")
    base_url = next(widget for widget in at.sidebar.text_input if widget.label == "Base URL")
    base_url.set_value("https://127.0.0.1/v1")
    at.run()
    primary = next(button for button in at.button if button.label.startswith("🚀"))
    primary.click().run()
    assert not at.exception
    assert any("Base URL" in str(err.value) or "alamat IP publik" in str(err.value) for err in at.error)


def test_app_renders_result_from_session_state():
    at = _fresh_app()
    at.session_state["clip"] = {
        "data": b"\x00\x00\x00\x20ftypisom" + b"\x00" * 256,
        "start_label": "00:01:00",
        "end_label": "00:01:45",
        "duration": 45.0,
        "title": "Momen uji",
        "reason": "uji render",
        "score": 80,
        "source_url": "https://youtu.be/abc123",
        "video_title": "Video uji",
        "model": "fake-model",
    }
    at.run()
    assert not at.exception
    assert at.subheader[0].value == "🎬 Hasil Klip"
    assert len(at.get("video")) == 1
    assert [item.label for item in at.get("download_button")] == [
        "⬇️ Download Klip SONYBAE (9:16)"
    ]
