"""SONYBAE CLIPS — antarmuka Streamlit.

UI saja. Semua logika pemrosesan ada di ``pipeline.py`` (bisa diuji tanpa server).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

import pipeline

APP_VERSION = "1.1.0"

MODE_SUBTITLE = "AI pilih momen (subtitle YouTube)"
MODE_VISUAL = "AI visual (tanpa subtitle)"
MODE_MANUAL_TRANSCRIPT = "AI + transkrip manual"
MODE_TIMESTAMP = "Timestamp manual"

MODE_CHOICES = [MODE_SUBTITLE, MODE_VISUAL, MODE_MANUAL_TRANSCRIPT, MODE_TIMESTAMP]
AI_MODES = (MODE_SUBTITLE, MODE_VISUAL, MODE_MANUAL_TRANSCRIPT)

PRESETS: dict[str, dict[str, str]] = {
    "OpenAI": {
        "base_url": "",
        "model": "gpt-4o-mini",
        "secret": "OPENAI_API_KEY",
        "note": "Kunci dari platform.openai.com",
    },
    "DeepSeek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "secret": "DEEPSEEK_API_KEY",
        "note": "Murah dan cepat untuk transkrip panjang",
    },
    "NVIDIA NIM": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "model": "meta/llama-3.3-70b-instruct",
        "secret": "NVIDIA_API_KEY",
        "note": "Gunakan model yang tersedia di akun NVIDIA NIM",
    },
    "OpenRouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "openai/gpt-4o-mini",
        "secret": "OPENROUTER_API_KEY",
        "note": "Satu kunci untuk banyak model",
    },
    "Custom / lainnya": {
        "base_url": "",
        "model": "gpt-3.5-turbo",
        "secret": "",
        "note": "Endpoint apa pun yang kompatibel dengan OpenAI",
    },
}

CSS = """
<style>
    .block-container {padding-top: 1.6rem; padding-bottom: 3rem;}
    div[data-testid="stButton"] > button,
    div[data-testid="stDownloadButton"] > button {width: 100%; font-weight: 600;}
    .sony-title {text-align:center; color:#ff2d2d; margin-bottom:.2rem;}
    .sony-sub {text-align:center; color:#8a8a8a; margin-top:0; font-size:.92rem;}
    footer {visibility:hidden;}
</style>
"""


def _secret(key: str, default: str | None = None) -> str | None:
    """Ambil nilai dari Streamlit Secrets tanpa meledak bila belum diset."""
    try:
        value = st.secrets.get(key, default)
    except Exception:  # noqa: BLE001 - secrets bisa tidak ada di server lokal
        return default
    return str(value) if value is not None else default


def _clear_credentials() -> None:
    """Callback tombol hapus; callback berjalan sebelum widget dibuat ulang."""
    for key in ("yt_cookies", *(f"api_key_{name}" for name in PRESETS)):
        st.session_state.pop(key, None)


def _cookie_file(job_path: Path) -> Path | None:
    """Simpan cookies YouTube (opsional) untuk mengatasi verifikasi bot."""
    raw = (st.session_state.get("yt_cookies") or _secret("YT_COOKIES") or "").strip()
    if not raw:
        return None
    path = job_path / "cookies.txt"
    path.write_text(raw + "\n", encoding="utf-8")
    return path


def _render_header() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("<h1 class='sony-title'>✂️ SONYBAE CLIPS</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='sony-sub'>Auto-clipping podcast &amp; live stream ke format 9:16 vertikal</p>",
        unsafe_allow_html=True,
    )
    st.divider()


def _render_sidebar() -> dict[str, Any]:
    with st.sidebar:
        st.header("🧠 Otak AI")
        provider = st.selectbox("Provider", list(PRESETS.keys()), index=0)
        preset = PRESETS[provider]
        st.caption(preset["note"])

        api_key = st.text_input(
            "API Key",
            type="password",
            key=f"api_key_{provider}",
            placeholder="sk-...",
            help="Dikirim ke provider terpilih; tersimpan di state sesi server sampai dihapus atau sesi berakhir.",
        )
        if provider == "Custom / lainnya":
            base_url = st.text_input(
                "Base URL",
                key="base_url_custom",
                placeholder="https://api.example.com/v1",
                help="Wajib HTTPS, port 443, hostname publik; gunakan API Key milik Anda sendiri.",
            )
        else:
            # Jangan izinkan pengguna mengganti endpoint sambil memakai API key dari Secrets.
            base_url = preset["base_url"]
            st.text_input(
                "Base URL",
                value=base_url or "Endpoint resmi OpenAI",
                disabled=True,
                key=f"base_url_fixed_{provider}",
                help="Endpoint tetap untuk mencegah API key server dikirim ke host lain.",
            )
        model_name = st.text_input(
            "Model",
            value=preset["model"],
            key=f"model_{provider}",
            help="Contoh: deepseek-chat, meta/llama-3.3-70b-instruct",
        )

        st.divider()
        with st.expander("🍪 Cookies YouTube (opsional)"):
            st.caption(
                "Isi bila muncul 'Sign in to confirm you're not a bot'. "
                "Format Netscape (hasil ekspor ekstensi 'Get cookies.txt'). "
                "Nilainya tetap di state sesi sampai dihapus atau sesi berakhir."
            )
            st.text_area(
                "Isi cookies.txt",
                key="yt_cookies",
                height=120,
                label_visibility="collapsed",
                placeholder="# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t0\tKEY\tVALUE",
            )

        st.button(
            "🧹 Hapus API key & cookies dari sesi",
            key="clear_credentials",
            on_click=_clear_credentials,
            help="Menghapus kredensial yang dimasukkan dari state sesi aplikasi.",
        )
        st.divider()
        st.caption(f"v{APP_VERSION} · YouTube → 9:16 · H.264/AAC")

        secret_key = preset.get("secret") or ""
        fallback = (_secret(secret_key) or "") if secret_key else ""
        if fallback:
            st.success("🔑 Memakai API key dari Secrets server.")

    return {
        "api_key": (api_key or "").strip() or fallback,
        "base_url": (base_url or "").strip(),
        "model": (model_name or "").strip(),
    }


def _render_result() -> None:
    clip = st.session_state.get("clip")
    if not clip:
        return

    st.divider()
    st.subheader("🎬 Hasil Klip")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Mulai", clip["start_label"])
    col_b.metric("Selesai", clip["end_label"])
    col_c.metric("Durasi", f"{clip['duration']:.0f} dtk")

    if clip.get("title"):
        st.markdown(f"**{clip['title']}**")
    if clip.get("reason"):
        st.info(clip["reason"])

    st.video(clip["data"])
    st.download_button(
        "⬇️ Download Klip SONYBAE (9:16)",
        data=clip["data"],
        file_name="sonybae_clip_9x16.mp4",
        mime="video/mp4",
        key="download_clip",
    )

    with st.expander("Detail teknis"):
        st.json(
            {
                "sumber": clip.get("source_url", ""),
                "judul_video": clip.get("video_title", ""),
                "timestamp": f"{clip['start_label']} - {clip['end_label']}",
                "ukuran_mb": round(len(clip["data"]) / (1024 * 1024), 2),
                "resolusi": f"{pipeline.TARGET_W}x{pipeline.TARGET_H}",
                "skor_ai": clip.get("score"),
                "model": clip.get("model", ""),
            }
        )

    if st.button("Bersihkan hasil", key="clear_clip"):
        st.session_state.pop("clip", None)
        st.rerun()


def _run_job(
    url: str,
    cfg: dict[str, Any],
    mode: str,
    clip_seconds: float,
    manual_start: str,
    manual_end: str,
    transcript_text: str = "",
) -> None:
    with pipeline.job_dir() as job:
        cookie_file = _cookie_file(job)

        st.write("Membaca metadata video...")
        meta = pipeline.fetch_metadata(url, cookie_file=cookie_file)
        if meta.is_live:
            raise pipeline.DownloadError(
                "Video ini masih live. Tunggu sampai selesai, lalu proses ulang."
            )
        if meta.title:
            st.write(f"🎙️ **{meta.title}**")
        if meta.duration:
            st.write(f"⏱️ Durasi video: {pipeline.format_timestamp(meta.duration)}")

        local_source: Path | None = None

        if mode == MODE_TIMESTAMP:
            start_s, end_s = pipeline.clamp_moment(
                manual_start, manual_end, meta.duration, min_seconds=5.0
            )
            moment = pipeline.ClipMoment(start=start_s, end=end_s)
            st.write(f"Potongan manual: {moment.start_label} - {moment.end_label}")
        elif mode == MODE_VISUAL:
            st.write("Mengunduh video sementara untuk analisis visual (tanpa subtitle)...")
            source_path = pipeline.download_source_video(
                url,
                job,
                expected_duration=meta.duration,
                cookie_file=cookie_file,
            )
            local_source = source_path
            moment = pipeline.analyze_video_frames(
                source_path,
                api_key=cfg["api_key"],
                model=cfg["model"],
                base_url=cfg["base_url"] or None,
                clip_seconds=clip_seconds,
                log=st.write,
            )
            st.write(
                f"✅ Momen visual terpilih: **{moment.start_label} - {moment.end_label}**"
                + (f" — {moment.title}" if moment.title else "")
            )
            if moment.reason:
                st.write(f"Alasan: {moment.reason}")
        elif mode == MODE_MANUAL_TRANSCRIPT:
            transcript = pipeline.normalize_transcript_text(transcript_text)
            st.write(
                f"Transkrip yang ditempel: {len(transcript.splitlines())} baris bertimestamp, "
                f"~{len(transcript):,} karakter."
            )
            moment = pipeline.analyze_transcript(
                transcript,
                api_key=cfg["api_key"],
                model=cfg["model"],
                base_url=cfg["base_url"] or None,
                clip_seconds=clip_seconds,
                duration=meta.duration,
                log=st.write,
            )
            st.write(
                f"✅ Momen terpilih: **{moment.start_label} - {moment.end_label}**"
                + (f" — {moment.title}" if moment.title else "")
            )
            if moment.reason:
                st.write(f"Alasan: {moment.reason}")
        else:
            st.write("Mengunduh transkrip subtitle dari YouTube...")
            raw_vtt = pipeline.download_transcript(url, job, cookie_file=cookie_file)
            cues = pipeline.parse_vtt(raw_vtt)
            transcript = pipeline.build_transcript_text(cues)
            st.write(
                f"Transkrip: {len(cues)} cue, ~{len(transcript):,} karakter "
                "(subtitle otomatis atau manual yang tersedia)."
            )
            moment = pipeline.analyze_transcript(
                transcript,
                api_key=cfg["api_key"],
                model=cfg["model"],
                base_url=cfg["base_url"] or None,
                clip_seconds=clip_seconds,
                duration=meta.duration,
                log=st.write,
            )
            st.write(
                f"✅ Momen terpilih: **{moment.start_label} - {moment.end_label}**"
                + (f" — {moment.title}" if moment.title else "")
            )
            if moment.reason:
                st.write(f"Alasan: {moment.reason}")

        if local_source is not None:
            final_path = pipeline.make_clip_from_local(
                local_source,
                moment.start,
                moment.end,
                job,
                min_seconds=pipeline.MIN_CLIP_SECONDS,
                max_seconds=clip_seconds,
                log=st.write,
            )
        else:
            final_path = pipeline.make_clip(
                url,
                moment.start,
                moment.end,
                job,
                cookie_file=cookie_file,
                duration=meta.duration,
                min_seconds=5.0 if mode == MODE_TIMESTAMP else pipeline.MIN_CLIP_SECONDS,
                max_seconds=(
                    clip_seconds if mode != MODE_TIMESTAMP else pipeline.MAX_CLIP_SECONDS
                ),
                log=st.write,
            )
        data = final_path.read_bytes()

    st.session_state["clip"] = {
        "data": data,
        "start_label": pipeline.format_timestamp(moment.start),
        "end_label": pipeline.format_timestamp(moment.end),
        "duration": moment.duration,
        "title": moment.title,
        "reason": moment.reason,
        "score": moment.score if mode != MODE_TIMESTAMP else None,
        "source_url": url,
        "video_title": meta.title,
        "model": cfg["model"] if mode != MODE_TIMESTAMP else "",
    }


def _process(
    url: str,
    cfg: dict[str, Any],
    mode: str,
    clip_seconds: float,
    manual_start: str,
    manual_end: str,
    transcript_text: str = "",
) -> None:
    st.session_state.pop("clip", None)
    pipeline.cleanup_stale_jobs()

    with st.status("Memproses klip SONYBAE...", expanded=True) as status:
        try:
            _run_job(
                url,
                cfg,
                mode,
                clip_seconds,
                manual_start,
                manual_end,
                transcript_text,
            )
        except Exception:
            status.update(label="Gagal memproses", state="error", expanded=True)
            raise
        status.update(label="Klip selesai dibuat", state="complete", expanded=False)


def _humanize_error(exc: Exception) -> str:
    message = str(exc) or exc.__class__.__name__
    if isinstance(exc, pipeline.TranscriptError):
        return message
    if isinstance(exc, pipeline.DownloadError):
        return f"{message}\n\n**Solusi:** isi Cookies YouTube di sidebar, lalu coba lagi."
    if isinstance(exc, pipeline.AIError):
        return f"{message}\n\n**Solusi:** cek API Key, Base URL, dan nama model di sidebar."
    if isinstance(exc, pipeline.SonybaeError):
        return message
    if isinstance(exc, ValueError):
        return message
    return f"Terjadi kesalahan tak terduga: {message}"


def main() -> None:
    st.set_page_config(
        page_title="SONYBAE CLIPS",
        page_icon="✂️",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    _render_header()
    cfg = _render_sidebar()

    missing = [name for name, ok in pipeline.check_binaries().items() if not ok]
    if missing:
        st.error(
            f"Server belum punya: {', '.join(missing)}. "
            "Pastikan packages.txt berisi 'ffmpeg' dan requirements.txt berisi 'yt-dlp'."
        )

    url = st.text_input(
        "🔗 Link YouTube Podcast / Live Stream",
        placeholder="https://www.youtube.com/watch?v=...",
    )

    mode = st.radio(
        "Mode potong",
        MODE_CHOICES,
        horizontal=True,
        help=(
            "Pilih AI visual tanpa subtitle untuk memilih berdasarkan gambar saja; "
            "butuh model AI vision/multimodal. Model ini tidak mendengar audio/dialogue."
        ),
    )

    transcript_text = ""
    if mode != MODE_TIMESTAMP:
        clip_seconds = float(st.slider("Target durasi klip (detik)", 20, 90, 45, step=5))
        manual_start = manual_end = ""
        if mode == MODE_MANUAL_TRANSCRIPT:
            transcript_text = st.text_area(
                "Transkrip bertimestamp (VTT/SRT atau [00:01:23] ucapan)",
                height=220,
                placeholder=(
                    "[00:01:23] Contoh ucapan pertama\n"
                    "[00:02:10] Contoh ucapan berikutnya"
                ),
                help=(
                    "Tempel subtitle dari transkrip yang Anda miliki/berhak gunakan. "
                    "Setiap baris teks biasa perlu diawali timestamp."
                ),
            )
    else:
        clip_seconds = 45.0
        col_start, col_end = st.columns(2)
        manual_start = col_start.text_input("Mulai (HH:MM:SS)", value="00:00:00")
        manual_end = col_end.text_input("Selesai (HH:MM:SS)", value="00:00:45")

    run_clicked = st.button("🚀 Proses Klip Viral SONYBAE!", type="primary")

    if mode == MODE_VISUAL:
        st.caption(
            f"Mode ini mengunduh sementara seluruh video (maks. {pipeline.MAX_VIDEO_ANALYSIS_MINUTES} menit / "
            f"{pipeline.MAX_VIDEO_ANALYSIS_BYTES // (1024 * 1024)} MB), mengirim frame sampel saja ke model vision, "
            "lalu memotong dari file lokal itu. File sumber dibuang setelah proses."
        )
    else:
        st.caption(
            "Server hanya mengunduh bagian yang dipilih (bukan video penuh), "
            "lalu crop ke 1080×1920. Hasil tidak disimpan permanen."
        )

    if run_clicked:
        try:
            clean_url = pipeline.validate_url(url)
            if mode in AI_MODES and not cfg["api_key"]:
                raise ValueError("API Key wajib diisi untuk mode AI.")
            if mode in AI_MODES and not cfg["model"]:
                raise ValueError("Nama model wajib diisi untuk mode AI.")
            if mode in AI_MODES and cfg["base_url"]:
                # Validasi lebih awal agar endpoint custom tidak dapat menjadi SSRF.
                pipeline.validate_base_url(cfg["base_url"])
            _process(
                clean_url,
                cfg,
                mode,
                clip_seconds,
                manual_start,
                manual_end,
                transcript_text,
            )
        except Exception as exc:  # noqa: BLE001 - semua error ditampilkan ke pengguna
            st.error(_humanize_error(exc))
            with st.expander("Log teknis"):
                st.code(repr(exc))

    _render_result()

    st.divider()
    st.caption(
        "SONYBAE CLIPS · untuk konten podcast/live stream vertikal. "
        "Hanya link YouTube yang diproses; API key hanya berada di state sesi sampai dihapus atau sesi berakhir."
    )


if __name__ == "__main__":
    main()
