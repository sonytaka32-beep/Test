"""Tes logika inti SONYBAE CLIPS (tanpa jaringan)."""

from __future__ import annotations

from pathlib import Path

import pytest

import pipeline


# --- Timestamp --------------------------------------------------------------


def test_parse_timestamp_formats():
    assert pipeline.parse_timestamp("00:01:02") == 62.0
    assert pipeline.parse_timestamp("01:02:03.500") == 3723.5
    assert pipeline.parse_timestamp("1:02:03") == 3723.0
    assert pipeline.parse_timestamp("02:03") == 123.0
    assert pipeline.parse_timestamp("12,25") == 12.25
    assert pipeline.parse_timestamp(12.5) == 12.5
    assert pipeline.parse_timestamp("  00:00:07 ") == 7.0


@pytest.mark.parametrize(
    "bad",
    ["", "abc", None, -3, "00:xx:01", float("nan"), float("inf"), "nan", "inf", "00:60:00", "00:00:60", "01:60:00"],
)
def test_parse_timestamp_rejects_invalid(bad):
    with pytest.raises(ValueError):
        pipeline.parse_timestamp(bad)


def test_format_timestamp():
    assert pipeline.format_timestamp(62) == "00:01:02"
    assert pipeline.format_timestamp(3723.5, millis=True) == "01:02:03.500"
    assert pipeline.format_timestamp(3723.5) == "01:02:03"
    assert pipeline.format_timestamp(-5) == "00:00:00"
    assert pipeline.format_timestamp(3600) == "01:00:00"


# --- VTT --------------------------------------------------------------------

SAMPLE_VTT = """WEBVTT
Kind: captions
Language: id

NOTE ini komentar

00:00:00.000 --> 00:00:02.000
baris satu

00:00:02.000 --> 00:00:04.000
baris satu
baris dua

00:00:04.000 --> 00:00:06.500
<00:00:04.200><c>baris</c> <c>dua</c> diteruskan

00:00:06.500 --> 00:00:09.000
&amp; selesai&nbsp;di sini
"""


def test_parse_vtt_basic():
    cues = pipeline.parse_vtt(SAMPLE_VTT)
    assert len(cues) >= 2
    assert cues[0].start == 0.0
    assert "baris satu" in cues[0].text
    assert all(cue.end > cue.start for cue in cues)
    assert any("&" in cue.text for cue in cues)
    assert all("<" not in cue.text and "&amp;" not in cue.text for cue in cues)


def test_parse_vtt_rolling_captions_are_merged():
    cues = pipeline.parse_vtt(SAMPLE_VTT)
    merged = pipeline.merge_cues(cues)
    assert len(merged) <= len(cues)
    transcript = pipeline.build_transcript_text(cues)
    assert transcript.startswith("[00:00:00]")
    assert transcript.count("baris satu") == 1
    assert "baris dua diteruskan" in transcript


def test_parse_vtt_empty_and_garbage():
    assert pipeline.parse_vtt("") == []
    assert pipeline.parse_vtt("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n") == []
    assert pipeline.parse_vtt("bukan subtitle sama sekali") == []


def test_fetch_metadata_retries_with_fallback_client_on_missing_formats(monkeypatch):
    calls = []

    class Result:
        def __init__(self, returncode, stdout="", stderr=""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if len(calls) == 1:
            return Result(
                1,
                stderr="ERROR: Requested format is not available. Use --list-formats",
            )
        return Result(0, "79.5\tVideo OK\tFalse\n")

    monkeypatch.setattr(pipeline, "_run", fake_run)
    meta = pipeline.fetch_metadata("https://youtu.be/abc123")

    assert len(calls) == 2
    first_client_arg = calls[0].index("--extractor-args")
    second_client_arg = calls[1].index("--extractor-args")
    assert calls[0][first_client_arg - 1] == "https://youtu.be/abc123"
    assert calls[1][second_client_arg - 1] == "https://youtu.be/abc123"
    assert calls[0][first_client_arg + 1] == f"youtube:player_client={pipeline.PLAYER_CLIENTS_PRIMARY}"
    assert calls[0][first_client_arg + 1] == "youtube:player_client=default"
    assert calls[1][second_client_arg + 1] == (
        f"youtube:player_client={pipeline.PLAYER_CLIENTS_FALLBACK}"
    )
    assert calls[1][second_client_arg + 1] == "youtube:player_client=default,web_embedded"
    assert meta.title == "Video OK"
    assert meta.duration == 79.5


def test_fetch_metadata_missing_formats_error_does_not_blame_cookies(monkeypatch):
    calls = []

    class Result:
        returncode = 1
        stdout = ""
        stderr = "ERROR: Requested format is not available."

    monkeypatch.setattr(pipeline, "_run", lambda cmd, **kwargs: (calls.append(cmd) or Result()))

    with pytest.raises(pipeline.DownloadError, match="tidak menyediakan format video") as exc:
        pipeline.fetch_metadata("https://youtu.be/abc123")

    assert len(calls) == 2
    assert "hapus dulu" in str(exc.value)
    assert "rusak" in str(exc.value)


def test_fetch_metadata_page_needs_reload_retries_with_fallback(monkeypatch):
    calls = []

    class Result:
        def __init__(self, returncode, stdout="", stderr=""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if len(calls) == 1:
            return Result(
                1,
                stderr="ERROR: The page needs to be reloaded.",
            )
        return Result(0, "79.5\tVideo OK\tFalse\n")

    monkeypatch.setattr(pipeline, "_run", fake_run)
    meta = pipeline.fetch_metadata("https://youtu.be/abc123")

    assert len(calls) == 2
    assert calls[0][calls[0].index("--extractor-args") + 1].endswith("default")
    assert calls[1][calls[1].index("--extractor-args") + 1] == (
        f"youtube:player_client={pipeline.PLAYER_CLIENTS_FALLBACK}"
    )
    assert meta.title == "Video OK"
    assert meta.duration == 79.5


def test_download_transcript_reports_rate_limit_separately(monkeypatch, tmp_path):
    calls = []

    class Result:
        returncode = 1
        stdout = ""
        stderr = "ERROR: Unable to download video subtitles: HTTP Error 429: Too Many Requests"

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return Result()

    monkeypatch.setattr(pipeline, "_run", fake_run)

    with pytest.raises(
        pipeline.TranscriptError,
        match="membatasi permintaan subtitle.*HTTP 429",
    ) as exc:
        pipeline.download_transcript("https://youtu.be/abc123", tmp_path)

    assert len(calls) == 1  # Jangan lanjutkan fallback yang menambah beban setelah 429.
    assert "Timestamp manual" in str(exc.value)
    assert "bukan masalah API key AI" in str(exc.value)
    assert "tidak punya subtitle" not in str(exc.value)


def test_normalize_pasted_timestamped_transcript():
    raw = "[00:01:23] Halo <b>semua</b>\n01:02:03.500 - Ucapan berikut"
    assert pipeline.normalize_transcript_text(raw) == (
        "[00:01:23] Halo semua\n[01:02:03] Ucapan berikut"
    )


def test_normalize_pasted_vtt_and_srt_transcripts():
    assert pipeline.normalize_transcript_text(SAMPLE_VTT).startswith("[00:00:00] baris satu")
    srt = "1\n00:00:02,000 --> 00:00:03,000\nIsi subtitle\n"
    assert pipeline.normalize_transcript_text(srt) == "[00:00:02] Isi subtitle"


@pytest.mark.parametrize("raw", ["", "teks tanpa waktu", "[00:99:00] invalid"])
def test_normalize_pasted_transcript_rejects_invalid_input(raw):
    with pytest.raises(pipeline.TranscriptError):
        pipeline.normalize_transcript_text(raw)


def test_build_transcript_has_seekable_timestamps():
    cues = [
        pipeline.Cue(0.0, 3.0, "awal"),
        pipeline.Cue(3725.0, 3728.0, "tengah podcast"),
    ]
    text = pipeline.build_transcript_text(cues)
    lines = text.splitlines()
    assert lines[0].startswith("[00:00:00] awal")
    assert lines[1].startswith("[01:02:05] tengah podcast")


# --- Chunking ---------------------------------------------------------------


def test_chunk_transcript_respects_budget_and_samples_video_span():
    lines = [f"[00:00:{i % 60:02d}] nomor-un {i:03d} " + ("kata " * 20) for i in range(200)]
    text = "\n".join(lines)

    chunks = pipeline.chunk_transcript(text, max_chars=2_000, max_chunks=3)
    assert 1 <= len(chunks) <= 3
    assert all(len(chunk) <= 2_000 for chunk in chunks)
    assert "nomor-un 000" in chunks[0]
    assert "nomor-un 199" in chunks[-1]
    assert any("nomor-un 100" in chunk for chunk in chunks)


def test_chunk_transcript_short_input_single_chunk():
    text = "\n".join(f"[00:00:{i:02d}] baris {i}" for i in range(10))
    assert len(pipeline.chunk_transcript(text)) == 1


def test_chunk_transcript_empty():
    assert pipeline.chunk_transcript("") == []
    assert pipeline.chunk_transcript("\n\n") == []


def test_chunk_transcript_single_chunk_samples_midpoint():
    text = "\n".join(f"[00:00:{i % 60:02d}] nomor {i:04d}" for i in range(500))
    chunks = pipeline.chunk_transcript(text, max_chars=100, max_chunks=1)
    assert len(chunks) == 1
    assert "nomor 0250" in chunks[0] or "nomor 0251" in chunks[0]


# --- JSON / kandidat AI -----------------------------------------------------


def test_extract_json_plain():
    assert pipeline.extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced_with_prose():
    raw = 'Berikut hasilnya:\n```json\n[{"start":"00:01:00","end":"00:01:45","skor":80}]\n```\nSemoga membantu.'
    data = pipeline.extract_json(raw)
    assert isinstance(data, list) and data[0]["start"] == "00:01:00"


def test_extract_json_braces_inside_string():
    raw = '{"judul": "kurung {aneh} di dalam", "start": "00:00:05"}'
    assert pipeline.extract_json(raw)["judul"] == "kurung {aneh} di dalam"


def test_extract_json_invalid_raises():
    with pytest.raises(ValueError):
        pipeline.extract_json("model menolak menjawab")


def test_coerce_candidates_shapes():
    single = {"clips": [{"start": "00:01:00", "end": "00:01:45", "judul": "A", "alasan": "B", "skor": 88}]}
    result = pipeline.coerce_candidates(single)
    assert result == [
        {"start": "00:01:00", "end": "00:01:45", "title": "A", "reason": "B", "score": 88}
    ]

    direct = pipeline.coerce_candidates({"start": "00:00:10", "end": "00:00:50"})
    assert len(direct) == 1 and direct[0]["score"] == 50

    mixed = pipeline.coerce_candidates(
        [{"start": "00:00:10", "end": "00:00:50", "score": 300}, {"judul": "tanpa waktu"}]
    )
    assert len(mixed) == 1 and mixed[0]["score"] == 100

    assert pipeline.coerce_candidates("teks") == []
    assert pipeline.coerce_candidates(None) == []


# --- Validasi klip ----------------------------------------------------------


def test_clamp_moment_min_and_max_duration():
    assert pipeline.clamp_moment("00:00:10", "00:00:20") == (10.0, 25.0)
    assert pipeline.clamp_moment("00:00:10", "00:00:50") == (10.0, 50.0)
    assert pipeline.clamp_moment("00:00:10", "00:05:00") == (10.0, 100.0)


def test_clamp_moment_respects_video_duration():
    start, end = pipeline.clamp_moment("00:09:50", "00:10:00", duration=600)
    assert end <= 600 and start < end


def test_clamp_moment_invalid_order():
    with pytest.raises(ValueError):
        pipeline.clamp_moment("00:01:00", "00:00:30")


def test_clamp_moment_start_past_end_of_video():
    with pytest.raises(ValueError):
        pipeline.clamp_moment("00:20:00", "00:20:30", duration=600)


def test_clamp_rejects_video_shorter_than_minimum():
    with pytest.raises(ValueError, match="lebih pendek"):
        pipeline.clamp_moment("00:00:00", "00:00:10", duration=10)


def test_manual_clip_bounds_are_preserved_at_pipeline_boundary(monkeypatch, tmp_path):
    seen = {}

    def fake_download(url, workdir, start, end, **kwargs):
        seen["range"] = (start, end)
        return tmp_path / "source.mp4"

    def fake_crop(source, destination):
        destination.write_bytes(b"fake mp4")
        return destination

    monkeypatch.setattr(pipeline, "download_section", fake_download)
    monkeypatch.setattr(pipeline, "crop_to_vertical", fake_crop)
    result = pipeline.make_clip(
        "https://youtu.be/abc123", 10, 15, tmp_path,
        duration=20, min_seconds=5, max_seconds=90,
    )
    assert seen["range"] == (10, 15)
    assert result.read_bytes() == b"fake mp4"


def test_crop_to_vertical_seeks_only_the_clip_range(monkeypatch, tmp_path):
    captured = {}

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        Path(cmd[-1]).write_bytes(b"x" * 30_000)
        return Result()

    monkeypatch.setattr(pipeline, "_run", fake_run)
    monkeypatch.setattr(
        pipeline,
        "probe_video",
        lambda path: {"width": 1920, "height": 1080, "duration": 600, "rotation": 0},
    )
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")

    pipeline.crop_to_vertical(source, tmp_path / "out.mp4", start=60.0, end=100.0)

    cmd = captured["cmd"]
    assert cmd[cmd.index("-ss") + 1] == "60.000"
    assert cmd[cmd.index("-t") + 1] == "40.000"
    assert cmd[cmd.index("-i") + 1] == str(source)


def test_make_clip_from_local_skips_second_download(monkeypatch, tmp_path):
    called = {}

    monkeypatch.setattr(
        pipeline,
        "probe_video",
        lambda path: {"width": 1280, "height": 720, "duration": 300, "rotation": 0},
    )

    def fake_crop(source, destination, **kwargs):
        called["crop"] = (source, kwargs)
        destination.write_bytes(b"mp4")
        return destination

    monkeypatch.setattr(pipeline, "crop_to_vertical", fake_crop)

    source = tmp_path / "visual_source.mp4"
    source.write_bytes(b"video")

    result = pipeline.make_clip_from_local(source, 10, 50, tmp_path)

    assert called["crop"][0] == source
    assert called["crop"][1] == {"start": 10.0, "end": 50.0}
    assert result.read_bytes() == b"mp4"


# --- URL --------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=abc123",
        "https://youtu.be/abc123",
        "https://m.youtube.com/watch?v=abc123",
        "http://youtube.com/live/abc123",
    ],
)
def test_validate_url_accepts_youtube(url):
    assert pipeline.validate_url(url) == url.strip()


@pytest.mark.parametrize(
    "url",
    ["", "   ", "ftp://youtube.com/watch?v=x", "https://evil.example.com/v", "youtube.com/watch?v=abc"],
)
def test_validate_url_rejects_others(url):
    with pytest.raises(ValueError):
        pipeline.validate_url(url)


# --- Filter ffmpeg ----------------------------------------------------------


def test_crop_filter_landscape_even_dimensions():
    vf = pipeline.build_crop_filter(1920, 1080)
    assert vf == "crop=606:1080:656:0,scale=1080:1920:flags=lanczos,setsar=1"


def test_crop_filter_small_landscape():
    vf = pipeline.build_crop_filter(1280, 720)
    assert vf.startswith("crop=404:720:438:0,")


def test_crop_filter_vertical_source_is_cropped_exactly():
    vf = pipeline.build_crop_filter(1080, 1920)
    assert vf.startswith("crop=1080:1920:0:0,")


def test_crop_filter_too_tall_source_uses_fit_and_crop():
    vf = pipeline.build_crop_filter(480, 1080)
    assert "force_original_aspect_ratio=increase" in vf
    assert vf.endswith("crop=1080:1920,setsar=1")


def test_crop_filter_odd_height_is_made_even():
    vf = pipeline.build_crop_filter(1920, 1081)
    assert vf.startswith("crop=606:1080:")


def test_target_constants():
    assert (pipeline.TARGET_W, pipeline.TARGET_H) == (1080, 1920)
    assert pipeline.TARGET_W * 16 == pipeline.TARGET_H * 9


# --- Analisis visual tanpa subtitle ----------------------------------------


def _fake_client(captured, content):
    from types import SimpleNamespace

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )

    return SimpleNamespace(chat=SimpleNamespace(completions=Completions()))


def test_chat_attaches_images_as_data_urls():
    captured = {}
    client = _fake_client(captured, '{"ok": true}')

    pipeline._chat(
        client,
        "vision-model",
        "pilih momen",
        temperature=0,
        images=["QUJD", "REVG"],
    )

    content = captured["messages"][1]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "pilih momen"}
    assert content[1]["image_url"]["url"] == "data:image/jpeg;base64,QUJD"
    assert content[2]["image_url"]["url"] == "data:image/jpeg;base64,REVG"


def test_chat_without_images_keeps_plain_string_message():
    captured = {}
    client = _fake_client(captured, '{"ok": true}')

    pipeline._chat(client, "text-model", "pilih momen", temperature=0)

    assert captured["messages"][1]["content"] == "pilih momen"


def test_analyze_video_frames_parses_and_clamps_ai_reply(monkeypatch):
    seen = {}

    monkeypatch.setattr(
        pipeline,
        "probe_video",
        lambda path: {"duration": 600.0, "width": 1280, "height": 720, "rotation": 0},
    )
    monkeypatch.setattr(
        pipeline, "_video_analysis_frames", lambda path: (["QUJD", "REVG"], [0.0, 300.0])
    )
    monkeypatch.setattr(pipeline, "_make_client", lambda key, base: "client")

    def fake_chat(client, model, prompt, **kwargs):
        seen["images"] = kwargs.get("images")
        seen["prompt"] = prompt
        return '{"start":"00:05:00","end":"00:05:40","judul":"Aksi","alasan":"gerakan jelas","skor":88}'

    monkeypatch.setattr(pipeline, "_chat", fake_chat)

    moment = pipeline.analyze_video_frames(
        Path("sample.mp4"), api_key="k", model="vision", clip_seconds=40
    )

    assert moment.start == 300.0
    assert moment.end == 340.0
    assert moment.title == "Aksi"
    assert moment.score == 88
    assert seen["images"] == ["QUJD", "REVG"]
    assert "tanpa" in seen["prompt"].lower() or "frame" in seen["prompt"].lower()


def test_analyze_video_frames_reports_unsupported_model(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "probe_video",
        lambda path: {"duration": 600.0, "width": 1280, "height": 720, "rotation": 0},
    )
    monkeypatch.setattr(
        pipeline, "_video_analysis_frames", lambda path: (["QUJD", "REVG"], [0.0, 300.0])
    )
    monkeypatch.setattr(pipeline, "_make_client", lambda key, base: "client")

    def boom(client, model, prompt, **kwargs):
        raise pipeline.AIError("Panggilan AI gagal: 400 unsupported content type")

    monkeypatch.setattr(pipeline, "_chat", boom)

    with pytest.raises(pipeline.AIError, match="vision"):
        pipeline.analyze_video_frames(Path("sample.mp4"), api_key="k", model="text-only")


def test_analyze_video_frames_rejects_long_video(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "probe_video",
        lambda path: {"duration": 3601.0, "width": 1280, "height": 720, "rotation": 0},
    )

    with pytest.raises(pipeline.EncodeError, match="maksimal"):
        pipeline.analyze_video_frames(Path("sample.mp4"), api_key="k", model="vision")


def test_video_analysis_frames_are_capped(monkeypatch, tmp_path):
    monkeypatch.setattr(
        pipeline,
        "probe_video",
        lambda path: {"duration": pipeline.MAX_VIDEO_ANALYSIS_MINUTES * 60.0,
                      "width": 1280, "height": 720, "rotation": 0},
    )
    calls = {"count": 0}

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, **kwargs):
        calls["count"] += 1
        Path(cmd[-1]).write_bytes(b"frame-bytes")
        return Result()

    monkeypatch.setattr(pipeline, "_run", fake_run)

    frames, times = pipeline._video_analysis_frames(tmp_path / "video.mp4")

    assert len(frames) == pipeline.VIDEO_ANALYSIS_MAX_FRAMES
    assert len(times) == len(frames)
    assert calls["count"] == pipeline.VIDEO_ANALYSIS_MAX_FRAMES
    assert all(0 <= timestamp <= pipeline.MAX_VIDEO_ANALYSIS_MINUTES * 60 for timestamp in times)
    assert times[0] > 0
    assert times[-1] > pipeline.MAX_VIDEO_ANALYSIS_MINUTES * 60 * 0.9
