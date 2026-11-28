# SONYBAE CLIPS

Aplikasi Streamlit untuk mencari momen dari video YouTube (subtitle, transkrip tempelan, atau analisis visual tanpa subtitle), memotong bagian itu saja, lalu mengekspor klip vertikal 9:16 (1080×1920, H.264/AAC).

## Fitur

- Kompatibel dengan endpoint OpenAI-style: OpenAI, DeepSeek, NVIDIA NIM, OpenRouter, dan endpoint kustom.
- AI memilih kandidat momen dari subtitle; tersedia AI + transkrip bertimestamp yang ditempel pengguna dan mode timestamp manual bila YouTube membatasi subtitle.
- **AI visual (tanpa subtitle)**: mengunduh video sementara, mengirim frame sampel ke model vision, lalu memotong klip dari file lokal tanpa transkrip/audio. Membutuhkan model multimodal.
- Mengunduh bagian terpilih saja via `yt-dlp --download-sections`, bukan seluruh video (kecuali mode AI visual).
- Crop tengah menjadi 1080×1920 dan encode MP4 yang siap diunduh.
- Direktori kerja sementara unik dan otomatis dihapus setelah tiap proses; hasil klip dibatasi 100 MB per unduhan.
- Mendukung cookies YouTube opsional untuk kondisi server terkena pemeriksaan bot.

## Cara menjalankan lokal

Perlu Python 3.10+ dan FFmpeg/FFprobe terpasang di sistem.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
python -m pip install pytest
streamlit run app.py
```

Buka URL lokal yang ditampilkan Streamlit. API key bisa dimasukkan di sidebar; atau simpan sebagai Streamlit Secret (lihat bagian konfigurasi rahasia).

## Deploy ke Streamlit Community Cloud

1. Buat repository GitHub baru atau unggah isi folder ini ke repository kosong.
2. Buka Streamlit Community Cloud dan pilih **Create app**.
3. Pilih repository, branch (umumnya `main`), dan file utama `app.py`.
4. Deploy. Streamlit membaca `requirements.txt`; `packages.txt` memasang FFmpeg sistem.
5. Bila diperlukan, buka **App settings → Secrets** dan masukkan salah satu contoh berikut:

```toml
# Pilih secret sesuai provider yang dipakai. Jangan commit file ini.
DEEPSEEK_API_KEY = "isi-di-dashboard"
# OPENAI_API_KEY = "..."
# NVIDIA_API_KEY = "..."
# OPENROUTER_API_KEY = "..."
```

Setelah deployment, masukkan link YouTube dan jalankan. Streamlit Community Cloud memakai resource terbatas dan layanan YouTube bisa membatasi unduhan dari IP pusat data; klip sangat panjang, video privat/berbayar, dan video live yang masih berlangsung tidak didukung.

## Checklist sebelum unggah ke GitHub

1. Jalankan `python -m pytest -q`; semua tes harus lulus.
2. Pastikan tidak ada `.streamlit/secrets.toml`, file `.mp4/.vtt`, atau `__pycache__` yang ikut ter-commit (sudah tercakup `.gitignore`).
3. Jangan menempelkan API key, cookies YouTube, atau token GitHub ke source, README, issue, atau URL remote.
4. Push ke repository baru (mis. `sonybae-clips-web`) pada branch `main`.
5. Hubungkan repository tersebut ke Streamlit Community Cloud seperti langkah di atas; setiap commit berikutnya otomatis memicu rebuild.

## Konfigurasi endpoint AI

Di sidebar, pilih provider atau **Custom / lainnya**, kemudian isi API Key, Base URL, dan model. Base URL sebaiknya endpoint tepercaya yang kompatibel dengan API chat-completions OpenAI. API key tidak ditulis ke log atau file proyek. Jika aplikasi dipakai publik dengan API key yang diatur oleh pemilik di Streamlit Secrets, preset memakai endpoint resmi tetap; jangan mengubah source agar key server dipasangkan dengan URL yang dapat dikendalikan pengunjung.

- OpenAI: Base URL kosong, contoh model `gpt-4o-mini`.
- DeepSeek: `https://api.deepseek.com/v1`, contoh `deepseek-chat`.
- NVIDIA NIM: `https://integrate.api.nvidia.com/v1`; pilih model yang tersedia di akun NIM Anda.
- OpenRouter: `https://openrouter.ai/api/v1`, nama model mengikuti katalog OpenRouter.
- Endpoint resmi pada preset dibuat tetap; opsi Custom hanya untuk API key pribadi dan mewajibkan HTTPS publik port 443.

Ketersediaan nama model, tarif, rate limit, dan syarat gratis ditentukan masing-masing penyedia dan dapat berubah. Endpoint custom divalidasi sebagai HTTPS port 443 yang resolve ke IP publik; untuk pertahanan SSRF menyeluruh, hosting publik tetap harus membatasi egress jaringan dan membatasi siapa yang boleh memakai aplikasi.

## Cookies YouTube (opsional)

Jika yt-dlp menampilkan *Sign in to confirm you're not a bot*, gunakan cookie Netscape yang diekspor dari akun Anda sendiri, lalu tempelkan ke ekspander cookies pada sidebar. Cookie hanya ditulis ke direktori sementara server selama pemrosesan dan dibuang saat selesai. Jangan tempel cookie ke repository, issue publik, atau log. Hindari menjalankan layanan ini untuk publik tanpa membatasi akses: cookie yang disediakan pengguna dikirim ke yt-dlp untuk permintaan YouTube.

## Uji

```bash
python -m pytest -q
```

Tes unit dan smoke-test UI tidak memanggil YouTube atau provider AI. GitHub Actions menjalankan tes untuk setiap pull request ke `main` dan setiap push ke `main`.

## Batasan dan tanggung jawab

- Hanya URL domain YouTube yang diterima.
- Mode AI memerlukan subtitle atau transkrip bertimestamp (format VTT/SRT atau baris `[HH:MM:SS] ucapan`); jika subtitle YouTube dibatasi, tempel transkrip sendiri atau pilih mode timestamp manual.
- Mode **AI visual (tanpa subtitle)** tidak memakai subtitle/audio sama sekali: momen dipilih dari frame gambar. Karena itu ia bisa melewatkan momen yang lucu/penting hanya dari ucapan. Batasnya video maksimal 12 menit dan 300 MB, dan model AI harus mendukung input gambar (vision); model teks saja akan ditolak dengan pesan jelas. Model vision yang tidak mendukung ini akan gagal dan aplikasi menyarankan mode lain.
- Unduhan klip memilih stream HTTPS (video+audio terpisah atau satu stream muxed) agar FFmpeg tidak mengambil potongan HLS langsung; jika YouTube tidak menyediakan format HTTPS untuk server tersebut, proses berhenti dengan pesan yang jelas. Kualitas maksimum bergantung pada format yang tersedia.
- Server YouTube bisa membatasi IP cloud atau mengubah cara akses subtitle/video. Error HTTP 429 berarti permintaan subtitle dibatasi YouTube, bukan berarti subtitle pasti tidak ada atau API key AI bermasalah. Error HTTP 403 saat unduh video berarti YouTube menolak akses stream dari server; aplikasi sudah mencoba mengeluarkan client `web_embedded` (yang dibatasi PO token) dan client cadangan, tetapi bila tetap 403, jangan mengulang berkali-kali—coba lagi setelah jeda atau pakai cookie milik akun sendiri.
- Video diunduh/diproses oleh server, sehingga perlu mempertimbangkan kuota, ruang sementara, batas ukuran upload/download, dan ketentuan layanan YouTube serta hak cipta. Gunakan hanya konten yang Anda miliki atau berhak proses.
- Waktu FFmpeg CPU dapat lama; batas Streamlit Community Cloud dapat berubah. Uji di hosting target sebelum membuka aplikasi untuk umum.
- Aplikasi ini tidak menyimpan arsip permanen. API key dan cookies yang dimasukkan di widget berada di state sesi server sampai dihapus dengan tombol sidebar atau sesi berakhir. Hasil klip juga tetap ada di state sesi server untuk preview/download sampai ditekan **Bersihkan hasil** atau sesi berakhir. Jangan gunakan kredensial YouTube sensitif pada aplikasi publik yang tidak Anda kendalikan.

## Struktur

```text
app.py                    # UI Streamlit
pipeline.py               # parsing VTT, analisis AI, yt-dlp, FFmpeg
requirements.txt          # dependensi Python (versi dipin)
packages.txt              # dependensi sistem untuk Streamlit Cloud
.streamlit/config.toml    # tema, telemetri, batas upload
.github/workflows/tests.yml
tests/                    # tes unit, keamanan, dan smoke test UI
```
