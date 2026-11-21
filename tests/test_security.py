"""Tes batas keamanan untuk endpoint dan unduhan."""

from __future__ import annotations

import socket

import pytest

import pipeline


def test_validate_base_url_accepts_public_https(monkeypatch):
    # Use a public resolver result without relying on external DNS in the test.
    monkeypatch.setattr(
        pipeline.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(None, None, None, None, ("8.8.8.8", 443))],
    )
    assert pipeline.validate_base_url("https://api.example.com/v1") == "https://api.example.com/v1"


@pytest.mark.parametrize(
    "url",
    [
        "http://api.example.com/v1",
        "https://127.0.0.1/v1",
        "https://localhost/v1",
        "https://169.254.169.254/latest/meta-data/",
        "https://user:pass@api.example.com/v1",
        "https://api.example.com:8443/v1",
        "https://api.example.com/v1?key=x",
    ],
)
def test_validate_base_url_rejects_unsafe_endpoint(url, monkeypatch):
    monkeypatch.setattr(
        pipeline.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(None, None, None, None, ("10.0.0.4", 443))],
    )
    with pytest.raises(pipeline.AIError):
        pipeline.validate_base_url(url)


def test_validate_base_url_rejects_private_dns(monkeypatch):
    monkeypatch.setattr(
        pipeline.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (None, None, None, None, ("8.8.8.8", 443)),
            (None, None, None, None, ("10.0.0.2", 443)),
        ],
    )
    with pytest.raises(pipeline.AIError, match="alamat IP publik"):
        pipeline.validate_base_url("https://mixed-dns.example/v1")


def test_validate_base_url_dns_failure(monkeypatch):
    def fail(*args, **kwargs):
        raise socket.gaierror("not found")

    monkeypatch.setattr(pipeline.socket, "getaddrinfo", fail)
    with pytest.raises(pipeline.AIError, match="di-resolve"):
        pipeline.validate_base_url("https://unresolved.example/v1")


def test_yt_dlp_format_fallback_is_height_limited(monkeypatch, tmp_path):
    captured = {}

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        (tmp_path / "source.mp4").write_bytes(b"x" * 60_000)
        return Result()

    monkeypatch.setattr(pipeline, "_run", fake_run)
    output = pipeline.download_section("https://youtu.be/abc123", tmp_path, 10, 20, max_height=720)
    fmt = captured["cmd"][captured["cmd"].index("-f") + 1]
    assert output.exists()
    assert all("height<=720" in alternative for alternative in fmt.split("/"))


def test_make_client_validates_before_constructing_openai(monkeypatch):
    called = {"value": False}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            called["value"] = True

    import openai

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    with pytest.raises(pipeline.AIError):
        pipeline._make_client("secret", "http://127.0.0.1/v1")
    assert not called["value"]
