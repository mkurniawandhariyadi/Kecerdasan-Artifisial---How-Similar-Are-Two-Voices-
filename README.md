# How Similar Are Two Voices?

Dashboard Streamlit: mikrofon realtime, perbandingan dua suara, graph & Dijkstra,
eksplorasi dataset, serta evaluasi. Paket memakai 160 rekaman dari empat speaker.

## File untuk GitHub

Unggah **isi folder ini** ke root repository, bukan seluruh folder proyek kuliah.

```text
.streamlit/config.toml
.gitignore
README.md
requirements.txt
packages.txt
streamlit_app.py
voice_demo_core.py
Export/                    # seluruh isi yang disertakan
```

Folder Export sudah lengkap: checkpoint, scaler MFCC, referensi speaker, threshold,
embedding, hasil evaluasi, graph, dan audio contoh. Jangan mengganti requirements.txt
root dengan requirements dari training Colab. Tidak membutuhkan notebook/Dataset asli.

## Push ke GitHub

Ada dua model berukuran sekitar 80–85 MiB. Upload melalui browser GitHub dibatasi
25 MiB per file; gunakan Git terminal atau fitur Publish to GitHub di VS Code.
Semua file paket masih di bawah batas Git biasa 100 MiB, jadi Git LFS tidak wajib.

Untuk repository GitHub BARU dan KOSONG, buka terminal dari folder ini:

```bash
git init
git add .
git commit -m "Add voice similarity dashboard"
git branch -M main
git remote add origin https://github.com/USERNAME/NAMA-REPO.git
git push -u origin main
```

Ganti USERNAME dan NAMA-REPO. Ikuti autentikasi GitHub yang diminta terminal.
ZIP hanya untuk memindahkan paket: ekstrak dulu, jangan upload ZIP sebagai aplikasinya.

## Deploy di Streamlit Community Cloud

1. Buka https://share.streamlit.io lalu Create app dan pilih repository GitHub.
2. Branch: main (atau branch tempat file Anda tersimpan).
3. Main file path: streamlit_app.py.
4. Advanced settings: pilih Python 3.12. Tidak ada secrets yang wajib diisi.
5. Deploy. requirements.txt memasang PyTorch/torchaudio CPU dan dependensi aplikasi;
   packages.txt memasang FFmpeg dan libsndfile untuk membaca audio.
6. Buka URL HTTPS aplikasi, pilih Mikrofon realtime, izinkan akses mikrofon,
   dan bicara setidaknya lima detik.

Mikrofon berjalan langsung melalui browser dan koneksi Streamlit, tanpa server
Gradio, port tambahan, atau API berbayar. Audio live tidak disimpan menjadi file.
Kinerja realtime bergantung pada CPU dan koneksi host; ini demo empat speaker dikenal.

## Lokal

```bash
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Sudah diuji lokal dengan Python 3.10. Deployment Cloud Python 3.12 belum dijalankan
pada saat paket disiapkan; jika gagal, lihat log build/runtime pada Manage app.
File watcher dinonaktifkan untuk menghindari import opsional SpeechBrain: setelah
mengedit kode lokal, hentikan dan jalankan ulang server.

## Referensi deployment

- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies
- https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository
