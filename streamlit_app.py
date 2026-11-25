"""Dashboard demo lokal: streamlit run streamlit_app.py."""
from pathlib import Path
import os
import time

import altair as alt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit.components.v2 import component

import voice_demo_core as core

st.set_page_config(page_title="How Similar Are Two Voices?", page_icon=":material/graphic_eq:", layout="wide")
ROOT = Path(__file__).parent
BUNDLE = Path(os.environ.get("VOICE_ARTIFACTS", ROOT / "Export"))
COLORS = ["#116B65", "#D27B40", "#5268A0", "#9874A0", "#B84A48", "#637C7E"]
METHOD_LABELS = {"MFCC + Cosine": "MFCC · cosine", "ECAPA Pretrained": "ECAPA · pretrained", "ECAPA Fine-Tuned": "ECAPA · fine-tuned"}


@st.cache_data(max_entries=3, show_spinner=False)
def read_bundle(path, stamp):
    return core.load_bundle(path)


@st.cache_data(max_entries=20, show_spinner=False)
def read_table(name, stamp):
    return pd.read_csv(BUNDLE / name)


@st.cache_resource(max_entries=4, show_spinner="Memuat encoder suara…")
def encoder(path, method, stamp):
    import torch
    torch.set_num_threads(2)
    return core.load_encoder(path, fine_tuned=method == core.METHODS[2])


@st.cache_resource(show_spinner="Menyiapkan pendeteksi ujaran…")
def vad_model():
    from silero_vad import load_silero_vad
    return load_silero_vad()


@st.cache_data(max_entries=9, show_spinner=False)
def gallery_for(method, stamp):
    return core.load_gallery(BUNDLE, method, CONFIG)


@st.cache_data(max_entries=24, show_spinner=False)
def graphs_for(method, k, stamp):
    return core.build_graph(MATRICES[method], MANIFEST, k)


@st.cache_data(max_entries=24, show_spinner=False)
def graph_positions(method, k, stamp):
    return nx.spring_layout(graphs_for(method, k, stamp)[1], seed=42, weight="layout_weight", iterations=90)


@st.cache_data(max_entries=6, show_spinner=False)
def projection(method, stamp):
    paths = list(MATRICES[method])
    matrix = np.stack([MATRICES[method][p] for p in paths])
    centered = matrix - matrix.mean(axis=0)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    result = MANIFEST.set_index("filepath").loc[paths, ["speaker", "session", "split", "filename"]].reset_index()
    result["PC1"], result["PC2"] = u[:, 0] * s[0], u[:, 1] * s[1]
    return result


def table(name):
    return read_table(name, (BUNDLE / name).stat().st_mtime_ns)


def section(title, description):
    st.title(title)
    st.markdown(description)


def score_chart(ranking, threshold):
    bars = alt.Chart(ranking).mark_bar(cornerRadiusEnd=5, size=26).encode(
        x=alt.X("Cosine:Q", scale=alt.Scale(domain=[-1, 1]), title="Cosine similarity"),
        y=alt.Y("Speaker:N", sort=alt.SortField("Cosine", order="descending"), title=None),
        color=alt.Color("Speaker:N", scale=alt.Scale(domain=SPEAKERS, range=COLORS[:len(SPEAKERS)]), legend=None),
        tooltip=["Speaker", alt.Tooltip("Cosine:Q", format=".4f"), "Lolos threshold:N"])
    rule = alt.Chart(pd.DataFrame({"Threshold": [threshold]})).mark_rule(color="#B84A48", strokeDash=[5, 4]).encode(x="Threshold:Q")
    return (bars + rule).properties(height=190)


def ranking_result(selected, ranking, threshold, actual=None):
    top = ranking.iloc[0]
    with st.container(horizontal=True, vertical_alignment="center", gap="large"):
        st.metric("Peringkat pertama", str(top.Speaker))
        st.metric("Cosine", f"{top.Cosine:.4f}")
        st.metric("Threshold", f"{threshold:.4f}")
    if selected:
        st.success(f"Suara paling dekat: {selected} · lolos threshold", icon=":material/check_circle:")
    else:
        st.warning(f"Paling dekat dengan {top.Speaker}, tetapi belum lolos threshold.", icon=":material/person_search:")
    if actual and actual != "Tidak diisi":
        st.caption(f"Cek label sebenarnya: {'sesuai' if selected == actual else 'belum sesuai'} · label ini tidak memengaruhi perhitungan.")
    st.altair_chart(score_chart(ranking, threshold), alt="Skor cosine setiap speaker dibanding threshold")
    st.dataframe(ranking, hide_index=True, column_config={
        "Cosine": st.column_config.NumberColumn(format="%.4f"),
        "Cosine × 100 (%)": st.column_config.NumberColumn(format="%.2f%%")}, alt="Ranking speaker dan keputusan threshold")


# CCv2 mempertahankan mikrofon browser saat fragment memperbarui hasil.
MICROPHONE = component("voice_live_microphone", html="""
<div class="mic-root">
  <div class="controls"><button id="start" type="button">Aktifkan mikrofon</button>
  <button id="stop" type="button" disabled>Berhenti</button>
  <button id="reconnect" type="button">Sambungkan ulang</button></div>
  <label for="device">Sumber mikrofon</label><select id="device" aria-label="Sumber mikrofon"><option value="">Perangkat default</option></select>
  <canvas id="wave" height="90" aria-label="Gelombang audio mikrofon secara langsung" role="img"></canvas>
  <p id="status" role="status" aria-live="polite">Aktifkan mikrofon dan izinkan akses di browser.</p>
</div>
""", css="""
.mic-root {font-family:var(--st-font); color:var(--st-text-color); padding:20px; background:var(--st-secondary-background-color); border-radius:12px;}
.controls {display:flex; flex-wrap:wrap; gap:8px; margin-bottom:16px;}
button,select {font:inherit; border:1px solid var(--st-border-color); border-radius:8px; padding:10px 15px; color:var(--st-text-color); background:var(--st-background-color); cursor:pointer;}
#start {background:var(--st-primary-color); color:white; border-color:var(--st-primary-color);}
button:hover:not(:disabled){filter:brightness(.93)} button:disabled {opacity:.48; cursor:default}
button:focus-visible,select:focus-visible {outline:3px solid var(--st-primary-color);outline-offset:3px;}
label {display:block;font-size:.9rem;margin-bottom:6px;} select {width:100%;}
canvas {width:100%;height:90px;display:block;margin-top:12px;} p{font-size:.9rem;margin:8px 0 0;min-height:24px;}
""", js="""
export default function(component) {
  const {parentElement, data, setStateValue} = component;
  const root = parentElement.querySelector('.mic-root');
  if (!root) return;
  if (!root.voice) root.voice = {active:false, seq:0, ack:0, ctx:null, stream:null, buffer:[], samples:0, sent:0};
  const s = root.voice;
  s.emit = setStateValue;
  s.ack = data?.ack ?? 0;
  const start=root.querySelector('#start'), stop=root.querySelector('#stop'), reconnect=root.querySelector('#reconnect');
  const status=root.querySelector('#status'), device=root.querySelector('#device'), canvas=root.querySelector('#wave');
  function idle(message, emit=true) {
    s.active=false; s.starting=false; s.cancelled=true;
    if(s.stream) s.stream.getTracks().forEach(t=>t.stop());
    if(s.node) {s.node.port.onmessage=null;s.node.disconnect();}
    if(s.ctx && s.ctx.state!=='closed') s.ctx.close();
    s.stream=null;s.ctx=null;s.buffer=[];s.samples=0;
    cancelAnimationFrame(s.frame); start.disabled=false;stop.disabled=true;device.disabled=false;
    status.textContent=message;
    if(emit) s.emit('packet',{active:false,id:s.id,seq:++s.seq,status:message});
  }
  async function activate() {
    if(s.active || s.starting) return;
    s.starting=true;s.cancelled=false;start.disabled=true;status.textContent='Meminta akses mikrofon…';
    try {
      if(!navigator.mediaDevices?.getUserMedia) throw new Error('Buka localhost atau HTTPS untuk mengakses mikrofon.');
      const stream=await navigator.mediaDevices.getUserMedia({video:false,audio:{channelCount:1,echoCancellation:true,noiseSuppression:true,autoGainControl:false,...(device.value?{deviceId:{exact:device.value}}:{})}});
      if(s.cancelled) {stream.getTracks().forEach(t=>t.stop());return;}
      s.stream=stream;s.ctx=new AudioContext({sampleRate:16000});await s.ctx.resume();
      const worklet=`class VoiceCapture extends AudioWorkletProcessor {constructor(){super();this.b=new Float32Array(4096);this.n=0;} process(inputs){const a=inputs[0]?.[0];if(a){for(let i=0;i<a.length;i++){this.b[this.n++]=a[i];if(this.n===this.b.length){this.port.postMessage(this.b);this.b=new Float32Array(4096);this.n=0;}}}return true;}} registerProcessor('voice-capture',VoiceCapture);`;
      const url=URL.createObjectURL(new Blob([worklet],{type:'application/javascript'}));
      try {await s.ctx.audioWorklet.addModule(url);} finally {URL.revokeObjectURL(url);}
      if(s.cancelled) return;
      s.node=new AudioWorkletNode(s.ctx,'voice-capture');
      s.source=s.ctx.createMediaStreamSource(stream);s.source.connect(s.node);
      const mute=s.ctx.createGain();mute.gain.value=0;s.node.connect(mute);mute.connect(s.ctx.destination);
      s.id=crypto.randomUUID();s.seq=0;s.ack=0;s.sent=0;s.pending=0;s.samples=0;s.buffer=[];s.active=true;s.starting=false;
      const max=Math.round(s.ctx.sampleRate*(data?.window_seconds??5));
      s.node.port.onmessage=event=>{
        if(!s.active) return;
        s.buffer.push(event.data);s.samples+=event.data.length;
        while(s.samples>max && s.buffer.length>1){const excess=s.samples-max;const first=s.buffer[0];if(first.length<=excess){s.buffer.shift();s.samples-=first.length;}else{s.buffer[0]=first.slice(excess);s.samples-=excess;}}
        const now=performance.now();
        // shortcut: satu paket menunggu ACK, sehingga inferensi lambat tidak menumpuk antrean.
        if(now-s.sent<1000 || (s.pending && s.ack<s.pending && now-s.sent<8000)) return;
        const pcm=new Int16Array(s.samples);let offset=0;
        for(const chunk of s.buffer){for(let i=0;i<chunk.length;i++) pcm[offset++]=Math.max(-32768,Math.min(32767,Math.round(chunk[i]*32768)));}
        const bytes=new Uint8Array(pcm.buffer);let binary='';
        for(let i=0;i<bytes.length;i+=8192) binary+=String.fromCharCode(...bytes.subarray(i,i+8192));
        s.pending=++s.seq;s.sent=now;
        s.emit('packet',{active:true,id:s.id,seq:s.seq,sample_rate:s.ctx.sampleRate,pcm:btoa(binary)});
        const duration=Math.min(5,s.samples/s.ctx.sampleRate);
        status.textContent=duration<5?'Mengumpulkan suara '+duration.toFixed(1)+' / 5 detik…':'Mikrofon aktif · skor diperbarui selama Anda berbicara.';
      };
      const analyser=s.ctx.createAnalyser();analyser.fftSize=1024;s.source.connect(analyser);
      const wave=new Float32Array(analyser.fftSize);
      function draw(){if(!s.active)return;canvas.width=Math.max(300,canvas.clientWidth*devicePixelRatio);canvas.height=90*devicePixelRatio;const c=canvas.getContext('2d');c.clearRect(0,0,canvas.width,canvas.height);analyser.getFloatTimeDomainData(wave);c.lineWidth=2*devicePixelRatio;c.strokeStyle=getComputedStyle(root).getPropertyValue('--st-primary-color')||'#116B65';c.beginPath();for(let i=0;i<wave.length;i++){const x=i/wave.length*canvas.width,y=(.5+wave[i]*1.8)*canvas.height;i?c.lineTo(x,y):c.moveTo(x,y);}c.stroke();s.frame=requestAnimationFrame(draw);}
      draw();start.disabled=true;stop.disabled=false;device.disabled=true;status.textContent='Mikrofon aktif · mengumpulkan jendela lima detik…';
      const selected=device.value;const devices=await navigator.mediaDevices.enumerateDevices();device.replaceChildren(new Option('Perangkat default',''));devices.filter(d=>d.kind==='audioinput').forEach((d,i)=>device.add(new Option(d.label||'Mikrofon '+(i+1),d.deviceId)));device.value=selected;
      stream.getAudioTracks()[0].onended=()=>idle('Mikrofon terputus. Klik Sambungkan ulang.');
    } catch(error) {idle(error.name==='NotAllowedError'?'Izin mikrofon ditolak. Izinkan mikrofon dari ikon pengaturan situs, lalu coba lagi.':'Mikrofon belum dapat digunakan: '+error.message);}
  }
  start.onclick=activate; stop.onclick=()=>idle('Mikrofon dihentikan. Klik Aktifkan mikrofon untuk memulai lagi.');
  reconnect.onclick=()=>{idle('Menyambungkan ulang…');activate();};
  return ()=>{queueMicrotask(()=>{if(!root.isConnected)idle('Mikrofon dihentikan.',false);});};
}
""")


def overview():
    section("How Similar Are Two Voices?", "Jelajahi karakter empat suara, bandingkan representasinya, lalu uji langsung dengan suara Anda.")
    with st.container(horizontal=True, gap="large"):
        st.badge("Export Colab dimuat", icon=":material/check_circle:", color="green")
        st.caption(f"{len(MANIFEST)} rekaman · {len(SPEAKERS)} speaker · 16 kHz · {METHOD_LABELS[METHOD]}")
    left, right = st.columns([1.7, 1], gap="large")
    with left:
        st.subheader("Empat suara dalam satu peta")
        pca = projection(METHOD, STAMP)
        chart = alt.Chart(pca).mark_circle(size=85, opacity=.8).encode(
            x=alt.X("PC1:Q", title="Komponen utama 1"), y=alt.Y("PC2:Q", title="Komponen utama 2"),
            color=alt.Color("speaker:N", title="Speaker", scale=alt.Scale(domain=SPEAKERS, range=COLORS[:len(SPEAKERS)])),
            shape=alt.Shape("split:N", title="Split"), tooltip=["speaker", "filename", "session", "split"]).interactive().properties(height=360)
        st.altair_chart(chart, alt="Proyeksi PCA embedding semua rekaman; warna menunjukkan speaker")
        st.caption("Setiap titik adalah satu rekaman. PCA merangkum vektor menjadi dua dimensi; jarak pada peta bukan skor cosine asli.")
    with right:
        st.subheader("Coba suaranya")
        sample_index = st.selectbox("Rekaman contoh", SAMPLES.index, format_func=lambda i: f"{SAMPLES.loc[i, 'speaker']} · {Path(SAMPLES.loc[i, 'source_filepath']).name}", key="overview_sample")
        row = SAMPLES.loc[sample_index]
        st.audio(sample_path(row).read_bytes(), alt=f"Rekaman contoh {row.speaker}")
        gallery = gallery_for(METHOD, STAMP)
        _, ranking = core.rank_speakers(MATRICES[METHOD][row.source_filepath], gallery, 1.)
        st.altair_chart(score_chart(ranking, 1.).layer[0], alt="Kemiripan contoh rekaman utuh dengan setiap referensi speaker")
        st.caption("Pratinjau memakai rekaman utuh. Demo realtime memakai jendela lima detik dan threshold tersendiri.")
        st.page_link(PAGES[1], label="Buka mikrofon realtime", icon=":material/mic:")
    st.subheader("Hasil test clean")
    comp = table("comparison.csv")
    st.dataframe(comp[["Model", "F1-score", "ROC-AUC", "EER"]], hide_index=True,
        column_config={x: st.column_config.NumberColumn(format="%.4f") for x in ["F1-score", "ROC-AUC", "EER"]}, alt="Hasil test clean ketiga metode")
    st.caption("Evaluasi ini untuk rekaman baru dari empat speaker dikenal. Nilai sempurna pada dataset kecil belum membuktikan generalisasi ke orang, hari, atau perangkat baru.")


def sample_path(row):
    path = (BUNDLE / row.audio_file).resolve()
    if not path.is_relative_to(BUNDLE.resolve()) or not path.is_file():
        raise ValueError("Audio contoh tidak tersedia di paket export.")
    return path


@st.fragment(run_every="1s")
def live_panel(threshold, actual):
    state = st.session_state.setdefault("live_state", {"token": None, "seq": 0, "history": [], "ranking": None, "status": "Aktifkan mikrofon untuk mulai."})
    packet_state = st.session_state.get("voice_microphone", {})
    packet = packet_state.get("packet") if packet_state else None
    result = MICROPHONE(key="voice_microphone", data={"ack": state["seq"], "window_seconds": WINDOW}, on_packet_change=lambda: None)
    packet = result.packet or packet
    if packet and (packet.get("id"), packet.get("seq"), METHOD, threshold) != state.get("processed"):
        token = packet.get("id")
        if token != state["token"] or state.get("method") != METHOD:
            state.update(token=token, history=[], started=time.monotonic(), ranking=None)
        state.update(seq=packet.get("seq", 0), processed=(token, packet.get("seq"), METHOD, threshold), method=METHOD,
                     received=time.monotonic(), selected=None, ranking=None)
        if not packet.get("active"):
            state["status"] = packet.get("status", "Mikrofon berhenti.")
        else:
            try:
                audio = core.decode_live_packet(packet, CONFIG["sample_rate"], WINDOW)
                duration = len(audio) / CONFIG["sample_rate"]
                if duration < WINDOW - .01:
                    state["status"] = f"Mengumpulkan suara: {duration:.1f} / {WINDOW:g} detik…"
                elif not core.contains_speech(audio, vad_model(), CONFIG):
                    state["status"] = "Menunggu ujaran · belum terdeteksi orang berbicara."
                else:
                    begin = time.perf_counter()
                    model = None if METHOD == core.METHODS[0] else encoder(str(BUNDLE), METHOD, STAMP)
                    vector = core.feature(core.prepare_live_audio(audio, CONFIG), METHOD, CONFIG, model)
                    selected, ranking = core.rank_speakers(vector, gallery_for(METHOD, STAMP), threshold)
                    state.update(selected=selected, ranking=ranking, status="Mikrofon aktif", latency=time.perf_counter()-begin)
                    elapsed = time.monotonic()-state["started"]
                    state["history"].extend({"Detik": elapsed, "Speaker": row.Speaker, "Cosine": row.Cosine} for row in ranking.itertuples())
                    state["history"] = state["history"][-240:]
            except (ValueError, RuntimeError, OSError, ImportError) as error:
                state["status"] = f"Audio belum dapat dihitung: {error}. Coba sambungkan ulang mikrofon."
    stale = packet and packet.get("active") and time.monotonic()-state.get("received", time.monotonic()) > 8
    if stale:
        state["ranking"] = None
        st.warning("Paket audio terhenti. Klik Sambungkan ulang; periksa koneksi dan izin mikrofon.")
    if state["ranking"] is not None:
        ranking_result(state["selected"], state["ranking"], threshold, actual)
        st.caption(f"Waktu komputasi terakhir {state.get('latency', 0)*1000:.0f} ms · jendela {WINDOW:g} detik · cosine × 100 bukan probabilitas identitas.")
    else:
        st.info(state["status"], icon=":material/mic:")
    if state["history"]:
        st.subheader("Pergerakan skor")
        history = pd.DataFrame(state["history"])
        line = alt.Chart(history).mark_line(point=False).encode(x=alt.X("Detik:Q", title="Waktu sejak mulai (detik)"),
            y=alt.Y("Cosine:Q", scale=alt.Scale(domain=[-1, 1])), color=alt.Color("Speaker:N", scale=alt.Scale(domain=SPEAKERS, range=COLORS[:len(SPEAKERS)])),
            tooltip=["Speaker", alt.Tooltip("Cosine:Q", format=".4f"), alt.Tooltip("Detik:Q", format=".1f")])
        rule = alt.Chart(pd.DataFrame({"Threshold": [threshold]})).mark_rule(color="#B84A48", strokeDash=[5, 4]).encode(y="Threshold:Q")
        st.altair_chart((line+rule).properties(height=230), alt="Riwayat cosine setiap speaker selama mikrofon aktif")


def microphone_page():
    section("Suara siapa yang paling dekat?", "Bicara secara langsung. Ranking dan skor bergerak selama mikrofon aktif, tanpa menyimpan file rekaman.")
    default = float(CONFIG["speaker_identification_demo"]["thresholds"][METHOD])
    left, right = st.columns([2, 1])
    with left:
        actual = st.selectbox("Label sebenarnya · opsional, hanya untuk cek", ["Tidak diisi"]+SPEAKERS, key="actual_speaker")
    with right:
        manual = st.toggle("Eksperimen threshold manual", key="manual_live_threshold")
    threshold = st.slider("Threshold demo sementara", -1., 1., default, .005, format="%.4f", key=f"live_threshold_{METHOD}") if manual else default
    st.caption(f"{METHOD_LABELS[METHOD]} · jendela {WINDOW:g} detik · threshold export {default:.4f}. " + ("Threshold manual hanya memengaruhi demo ini." if manual else "Threshold dikalibrasi pada validation, bukan test."))
    live_panel(threshold, actual)
    with st.expander("Cara membaca hasil", icon=":material/help:"):
        st.markdown("**Peringkat pertama** adalah suara referensi paling dekat. Label diterima hanya bila cosine melewati threshold. Skor persen adalah **cosine × 100**, bukan peluang identitas dan tidak berjumlah 100%. Saat diam, keputusan dikosongkan oleh pendeteksi ujaran.")
        st.caption("Demo penolakan orang di luar empat anggota belum divalidasi. Mengubah threshold dapat menaikkan penerimaan sekaligus risiko salah menerima.")


def audio_control(label, key):
    source = st.segmented_control(f"Sumber audio {label}", ["Contoh test", "Unggah", "Rekam"], default="Contoh test", required=True, key=key+"_source")
    if source == "Contoh test":
        selected = st.selectbox(f"Rekaman {label}", SAMPLES.index, index=0 if label=="A" else min(1, len(SAMPLES)-1),
            format_func=lambda i: f"{SAMPLES.loc[i, 'speaker']} · {Path(SAMPLES.loc[i, 'source_filepath']).name}", key=key+"_example")
        path = sample_path(SAMPLES.loc[selected]);data, name = path.read_bytes(), path.name
    else:
        file = st.file_uploader(f"Pilih audio {label}", type=["wav", "m4a", "ogg", "mp3", "flac"], key=key) if source=="Unggah" else st.audio_input(f"Rekam audio {label}", key=key+"_mic")
        if file is None:return None
        data, name = file.getvalue(), file.name
    mime = {".m4a":"audio/mp4", ".ogg":"audio/ogg", ".mp3":"audio/mpeg", ".flac":"audio/flac"}.get(Path(name).suffix.lower(),"audio/wav")
    st.audio(data, format=mime, alt=f"Audio {label}: {name}")
    return data, name


def compare_page():
    section("Bandingkan dua suara", "Satu pasangan, satu skor cosine. Coba orang yang sama, orang berbeda, lalu tambahkan noise.")
    a, b = st.columns(2, gap="large")
    with a:
        st.subheader("Audio A");input_a=audio_control("A", "compare_a")
    with b:
        st.subheader("Audio B");input_b=audio_control("B", "compare_b")
    noise = st.toggle("Tambahkan noise pada audio B", key="compare_noise")
    snr = st.slider("SNR · lebih kecil berarti lebih bising (dB)", 0, 30, 10, key="compare_snr") if noise else None
    signature = (METHOD, input_a, input_b, snr)
    if st.button("Hitung kemiripan", type="primary", icon=":material/compare_arrows:", disabled=input_a is None or input_b is None):
        try:
            with st.spinner("Menghitung representasi kedua suara…"):
                aa, bb = [core.read_audio(*item, CONFIG["sample_rate"], CONFIG["trim_top_db"]) for item in [input_a, input_b]]
                if noise:bb=core.noisy(bb, snr, CONFIG["seed"])
                model=None if METHOD==core.METHODS[0] else encoder(str(BUNDLE), METHOD, STAMP)
                va,vb=[core.feature(x,METHOD,CONFIG,model) for x in [aa,bb]]
                st.session_state["comparison"]={"method":METHOD,"score":float(np.clip(va@vb,-1,1)),"vector_a":va,"vector_b":vb,"name_a":input_a[1],"name_b":input_b[1],"signature":signature}
        except (ValueError,RuntimeError,OSError,ImportError) as error:st.error(f"Audio belum dapat dihitung: {error}")
    current=st.session_state.get("comparison")
    if not current or current["signature"]!=signature:return
    score, threshold=current["score"],CONFIG["thresholds"][METHOD]
    with st.container(horizontal=True,gap="large"):
        st.metric("Cosine similarity",f"{score:.4f}");st.metric("Threshold pasangan",f"{threshold:.4f}")
    (st.success if score>=threshold else st.warning)("Genuine · speaker sama menurut threshold" if score>=threshold else "Impostor · speaker berbeda menurut threshold")
    subset=REFERENCE[REFERENCE.model==METHOD].copy();subset["Jenis"]=subset.label.map({0:"Impostor",1:"Genuine"})
    hist=alt.Chart(subset).mark_bar(opacity=.65).encode(x=alt.X("score:Q",bin=alt.Bin(maxbins=25),title="Cosine similarity"),y=alt.Y("count():Q",stack=None,title="Pasangan test"),color=alt.Color("Jenis:N",scale=alt.Scale(domain=["Genuine","Impostor"],range=[COLORS[0],COLORS[1]])))
    marks=alt.Chart(pd.DataFrame({"Skor":[score,threshold],"Penanda":["Input Anda","Threshold"]})).mark_rule(strokeWidth=2).encode(x="Skor:Q",strokeDash="Penanda:N")
    st.altair_chart((hist+marks).properties(height=280),alt="Distribusi skor genuine dan impostor serta posisi skor pasangan input")
    st.caption("Threshold pasangan berbeda dari threshold query–referensi pada menu mikrofon. Keduanya dipilih dari validation untuk tugas yang berbeda.")
    st.page_link(PAGES[3],label="Telusuri pasangan ini di graph",icon=":material/route:")


def network_figure(graph, positions, source, target, path, local=False, show_labels=False):
    visible=set(graph)
    if local:
        visible={source,target,*path};visible.update(graph.neighbors(source));visible.update(graph.neighbors(target))
    view=graph.subgraph(visible);fig=go.Figure()
    for bridge,color,dash in [(False,"#C8D4CF","solid"),(True,"#C58340","dash")]:
        x,y=[],[]
        for a,b,d in view.edges(data=True):
            if d["mst_bridge"]==bridge:x.extend([positions[a][0],positions[b][0],None]);y.extend([positions[a][1],positions[b][1],None])
        fig.add_trace(go.Scatter(x=x,y=y,mode="lines",line=dict(color=color,width=1,dash=dash),hoverinfo="skip",showlegend=False))
    for a,b in zip(path,path[1:]):
        fig.add_trace(go.Scatter(x=[positions[a][0],positions[b][0]],y=[positions[a][1],positions[b][1]],mode="lines",line=dict(color="#B84A48",width=4,dash="dash" if graph[a][b]["mst_bridge"] else "solid"),hoverinfo="text",text=f"{graph.nodes[a]['node_id']} → {graph.nodes[b]['node_id']} · jarak {graph[a][b]['weight']:.4f}",showlegend=False))
    for i,speaker in enumerate(sorted({graph.nodes[p]["speaker"] for p in graph})):
        nodes=[p for p in view if view.nodes[p]["speaker"]==speaker]
        fig.add_trace(go.Scatter(x=[positions[p][0] for p in nodes],y=[positions[p][1] for p in nodes],mode="markers+text" if show_labels else "markers",name=speaker,
            text=[view.nodes[p]["node_id"] for p in nodes],textposition="top center",textfont=dict(size=10),
            customdata=[[view.nodes[p]["node_id"],speaker,Path(p).name] for p in nodes],hovertemplate="%{customdata[0]} · %{customdata[1]}<br>%{customdata[2]}<extra></extra>",
            marker=dict(size=[18 if p in [source,target] else 9 for p in nodes],color=COLORS[i%len(COLORS)],symbol=["square" if p==source else "diamond" if p==target else "circle" for p in nodes],line=dict(width=1,color="white"))))
    fig.update_layout(height=530,margin=dict(l=10,r=10,t=15,b=10),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),yaxis=dict(visible=False,scaleanchor="x",scaleratio=1),legend=dict(orientation="h",y=1.06),dragmode="pan",uirevision=str(sorted(view)),font=dict(color="#172D35"))
    return fig


def graph_page():
    section("Graph & Dijkstra", "Lihat hubungan seluruh rekaman, lalu telusuri jalur terpendek dengan bobot 1 − cosine.")
    controls=st.columns([1,1,1])
    with controls[0]:k=st.slider("Tetangga k-NN",1,8,3,key="graph_k")
    with controls[1]:mst=st.toggle("Tambahkan jembatan MST",key="graph_mst",help="Pembanding untuk menghubungkan komponen. Graph utama tetap k-NN murni.")
    with controls[2]:local=st.toggle("Fokus sekitar jalur",key="graph_local")
    natural,joined=graphs_for(METHOD,k,STAMP);positions=graph_positions(METHOD,k,STAMP)
    current=st.session_state.get("comparison")
    use_pair=False
    if current and current["method"]==METHOD:
        use_pair=st.toggle("Masukkan Audio A dan B dari perbandingan terakhir",key="graph_use_pair")
    if use_pair:
        vectors=dict(MATRICES[METHOD]);vectors.update(QA=current["vector_a"],QB=current["vector_b"])
        natural,joined=core.build_graph(vectors,MANIFEST,k);positions=nx.spring_layout(joined,seed=42,weight="layout_weight",iterations=90)
    graph=joined if mst else natural;nodes=sorted(graph,key=lambda p:(p not in ["QA","QB"],graph.nodes[p]["node_id"]))
    formatter=lambda p:f"{graph.nodes[p]['node_id']} · {graph.nodes[p]['speaker']} · {Path(p).name}"
    left,right=st.columns(2)
    with left:source=st.selectbox("Source · node awal",nodes,index=nodes.index("QA") if use_pair else 0,format_func=formatter,key="graph_source")
    with right:target=st.selectbox("Target · node tujuan",nodes,index=nodes.index("QB") if use_pair else min(7,len(nodes)-1),format_func=formatter,key="graph_target")
    result=core.path_result(graph,source,target)
    with st.container(horizontal=True,gap="large"):
        st.metric("Node",graph.number_of_nodes());st.metric("Edge",graph.number_of_edges());st.metric("Komponen",nx.number_connected_components(graph))
        st.metric("Jarak jalur",f"{result['distance']:.4f}" if result["path"] else "Tidak ada jalur")
    if result["path"]:
        st.caption(" → ".join(graph.nodes[p]["node_id"] for p in result["path"]))
        steps=st.slider("Telusuri langkah Dijkstra",0,result["hops"],result["hops"],key=f"graph_steps_{METHOD}_{k}_{mst}_{source}_{target}") if result["hops"] else 0
        shown_path=result["path"][:steps+1]
    else:
        st.info("Source dan target berada pada komponen berbeda. Ini hasil sah pada k-NN utama. Aktifkan MST untuk melihat jalur pembanding.")
        shown_path=[]
    labels=st.checkbox("Tampilkan ID semua node",value=local,key="graph_labels")
    st.plotly_chart(network_figure(graph,positions,source,target,shown_path,local,labels),config={"scrollZoom":True,"displaylogo":False},alt="Graph kemiripan suara; jalur Dijkstra merah, jembatan MST putus-putus")
    st.caption("Warna = speaker · kotak = source · wajik = target · merah = jalur yang ditelusuri. Geser, zoom, atau arahkan kursor ke node untuk membaca rekamannya.")
    st.caption("Slider menyorot urutan edge pada jalur hasil Dijkstra, bukan urutan semua node yang diperiksa algoritma.")
    if result["path"]:
        st.subheader("Bobot setiap langkah")
        frame=core.path_table(graph,result["path"])
        st.dataframe(frame,hide_index=True,alt="Rincian bobot edge dan total kumulatif jalur Dijkstra")
        st.caption(f"{result['hops']} edge · {result['mst_edges']} jembatan MST. Total jalur adalah penjumlahan jarak edge; bukan cosine langsung atau probabilitas identitas.")
        st.download_button("Unduh tabel jalur",frame.to_csv(index=False),"jalur_dijkstra.csv","text/csv",icon=":material/download:")
    with st.expander("Graph hasil export dan analisis semua pasangan"):
        st.dataframe(table("graph_path_summary.csv"),hide_index=True,alt="Keterhubungan genuine dan impostor pada graph hasil notebook")
        st.caption("Ringkasan export memakai ECAPA fine-tuned dan k=3 pada seluruh split untuk eksplorasi. Kontrol di atas menghitung ulang graph demo sesuai metode terpilih.")
        st.download_button("Unduh graph yang tampil", "\n".join(nx.generate_graphml(graph)),"similarity_demo.graphml","application/graphml+xml")


def dataset_page():
    section("Eksplorasi dataset", "Kenali data di balik skor: siapa yang berbicara, sesi rekaman, dan pembagian eksperimen.")
    chosen=st.multiselect("Speaker",SPEAKERS,default=SPEAKERS,key="dataset_speakers")
    split=st.pills("Split",["train","validation","test","stress"],default=["train","validation","test","stress"],selection_mode="multi",key="dataset_split")
    subset=MANIFEST[MANIFEST.speaker.isin(chosen)&MANIFEST.split.isin(split)]
    with st.container(horizontal=True,gap="large"):
        st.metric("Rekaman terpilih",len(subset));st.metric("Speaker",subset.speaker.nunique());st.metric("Total durasi",f"{subset.duration.sum()/60:.1f} menit")
    if subset.empty:st.info("Pilih setidaknya satu speaker dan split.");return
    left,right=st.columns(2,gap="large")
    with left:
        st.subheader("Pembagian rekaman")
        chart=alt.Chart(subset).mark_bar(cornerRadiusEnd=3).encode(x=alt.X("speaker:N",title="Speaker"),y=alt.Y("count():Q",title="Rekaman"),color=alt.Color("split:N",title="Split"),tooltip=["speaker","split","count()"])
        st.altair_chart(chart.properties(height=270),alt="Jumlah rekaman setiap speaker berdasarkan split")
    with right:
        st.subheader("Durasi tiap sesi")
        chart=alt.Chart(subset).mark_circle(size=70,opacity=.7).encode(x=alt.X("session:N",title="Sesi"),y=alt.Y("duration:Q",title="Durasi (detik)"),color=alt.Color("speaker:N",scale=alt.Scale(domain=SPEAKERS,range=COLORS[:len(SPEAKERS)])),tooltip=["speaker","filename",alt.Tooltip("duration:Q",format=".2f"),"split"])
        st.altair_chart(chart.properties(height=270),alt="Durasi rekaman untuk sesi A, B, C, D dan rekaman lama L")
    st.caption("A + rekaman lama L → training · B → validation/threshold · C → test · D → uji tambahan jarak dan naskah baru.")
    st.dataframe(subset[["speaker","filename","session","split","duration","original_sr","original_channels"]],hide_index=True,alt="Manifest rekaman yang telah difilter")
    st.download_button("Unduh manifest terpilih",subset.to_csv(index=False),"manifest_terpilih.csv","text/csv",icon=":material/download:")
    st.subheader("Dengarkan dan lihat sinyal")
    example=st.selectbox("Audio test untuk inspeksi",SAMPLES.index,format_func=lambda i:f"{SAMPLES.loc[i,'speaker']} · {Path(SAMPLES.loc[i,'source_filepath']).name}",key="inspect_sample")
    row=SAMPLES.loc[example];path=sample_path(row);st.audio(path.read_bytes(),alt=f"Audio inspeksi {row.speaker}")
    import soundfile as sf
    audio,sr=sf.read(path);step=max(1,len(audio)//2000)
    st.line_chart(pd.DataFrame({"Detik":np.arange(0,len(audio),step)/sr,"Amplitudo":audio[::step]}),x="Detik",y="Amplitudo",height=200,alt="Gelombang audio contoh berdasarkan waktu")
    if st.toggle("Lihat spektrogram",key="show_spectrogram"):
        import librosa
        spec=librosa.amplitude_to_db(np.abs(librosa.stft(audio.astype(np.float32),n_fft=512,hop_length=160)),ref=np.max)
        fig=go.Figure(go.Heatmap(z=spec,x=np.arange(spec.shape[1])*160/sr,y=librosa.fft_frequencies(sr=sr,n_fft=512),colorscale="Viridis",colorbar=dict(title="dB")))
        fig.update_layout(height=280,xaxis_title="Detik",yaxis_title="Frekuensi (Hz)",margin=dict(t=10,b=30,l=40,r=10))
        st.plotly_chart(fig,alt="Spektrogram frekuensi dan energi audio contoh")


def evaluation_page():
    section("Evaluasi & ketahanan", "Bandingkan metode dari hasil export, lalu lihat pengaruh threshold pada genuine dan impostor.")
    comparison=table("comparison.csv")
    metric=st.segmented_control("Metrik perbandingan",["F1-score","ROC-AUC","EER","FAR","FRR"],default="F1-score",required=True,key="evaluation_metric")
    chart=alt.Chart(comparison).mark_bar(cornerRadiusEnd=5,size=45).encode(x=alt.X(f"{metric}:Q",scale=alt.Scale(domain=[0,1])),y=alt.Y("Model:N",title=None),color=alt.Color("Model:N",legend=None),tooltip=["Model",alt.Tooltip(f"{metric}:Q",format=".4f")]).properties(height=200)
    st.altair_chart(chart,alt=f"Perbandingan {metric} ketiga metode pada test clean")
    st.caption("F1 dan ROC-AUC: lebih tinggi lebih baik. EER, FAR, FRR: lebih rendah lebih baik.")
    st.dataframe(comparison,hide_index=True,alt="Seluruh metrik test clean pada export Colab")
    st.subheader("Apa yang berubah saat threshold digeser?")
    default=float(CONFIG["thresholds"][METHOD]);threshold=st.slider("Threshold simulasi pada skor test",-1.,1.,default,.005,format="%.4f",key=f"evaluation_threshold_{METHOD}")
    subset=REFERENCE[REFERENCE.model==METHOD];labels=subset.label.to_numpy();scores=subset.score.to_numpy();prediction=scores>=threshold
    tn=int(((labels==0)&~prediction).sum());fp=int(((labels==0)&prediction).sum());fn=int(((labels==1)&~prediction).sum());tp=int(((labels==1)&prediction).sum())
    left,right=st.columns([1,1.8])
    with left:
        cm=pd.DataFrame({"Aktual":["Impostor","Impostor","Genuine","Genuine"],"Keputusan":["Impostor","Genuine","Impostor","Genuine"],"Pasangan":[tn,fp,fn,tp]})
        heat=alt.Chart(cm).mark_rect().encode(x="Keputusan:N",y="Aktual:N",color=alt.Color("Pasangan:Q",scale=alt.Scale(scheme="teals"),legend=None))
        texts=alt.Chart(cm).mark_text(fontSize=20).encode(x="Keputusan:N",y="Aktual:N",text="Pasangan:Q",color=alt.condition(alt.datum.Pasangan>max(tn,fp,fn,tp)*.45,alt.value("white"),alt.value("#172D35")))
        st.altair_chart((heat+texts).properties(height=220),alt="Confusion matrix genuine dan impostor pada threshold simulasi")
        st.caption(f"FAR {fp/max(1,fp+tn):.2%} · FRR {fn/max(1,fn+tp):.2%}")
    with right:
        from sklearn.metrics import roc_curve
        fpr,tpr,_=roc_curve(labels,scores)
        roc=pd.DataFrame({"FAR":fpr,"True positive rate":tpr})
        curve=alt.Chart(roc).mark_line(color=COLORS[0]).encode(x=alt.X("FAR:Q",scale=alt.Scale(domain=[0,1])),y=alt.Y("True positive rate:Q",scale=alt.Scale(domain=[0,1])))
        point=alt.Chart(pd.DataFrame({"FAR":[fp/max(1,fp+tn)],"True positive rate":[tp/max(1,tp+fn)]})).mark_point(color="#B84A48",filled=True,size=100).encode(x="FAR:Q",y="True positive rate:Q")
        st.altair_chart((curve+point).properties(height=220),alt="ROC curve dengan titik operasi threshold simulasi")
    st.caption("Slider ini hanya simulasi pada skor test. Threshold resmi tetap yang dipilih pada validation; hasil export tidak diubah.")
    st.subheader("Uji noise dan perubahan kondisi")
    robust=table("robustness.csv")
    order=robust.Condition.drop_duplicates().tolist()
    chart=alt.Chart(robust).mark_line(point=True).encode(x=alt.X("Condition:N",sort=order,title="Kondisi"),y=alt.Y("F1-score:Q",scale=alt.Scale(domain=[0,1])),color="Model:N",tooltip=["Model","Condition",alt.Tooltip("F1-score:Q",format=".4f")]).properties(height=270)
    st.altair_chart(chart,alt="Perubahan F1 ketiga metode pada clean, noise, dan uji lintas sesi")
    st.caption("Uji C–D mengubah jarak dan naskah sekaligus. Perubahan hasil tidak dapat dianggap sebagai pengaruh jarak saja.")
    with st.expander("Rincian robustness, kasus salah, dan konfigurasi"):
        st.dataframe(robust,hide_index=True,alt="Metrik tiap metode dan kondisi uji")
        failures=table("failure_pairs.csv")
        st.dataframe(failures,hide_index=True,alt="Contoh pasangan yang salah menurut threshold")
        st.json({"split":CONFIG["actual_split_counts"],"threshold_pasangan":CONFIG["thresholds"],"threshold_demo":CONFIG["speaker_identification_demo"]["thresholds"],"fine_tuning":CONFIG["fine_tuning"]})
    st.download_button("Unduh hasil evaluasi",comparison.to_csv(index=False),"comparison.csv","text/csv",icon=":material/download:")


try:
    STAMP=(BUNDLE/"config.json").stat().st_mtime_ns
    CONFIG,MANIFEST,MATRICES,REFERENCE=read_bundle(str(BUNDLE),STAMP)
    SAMPLES=pd.read_csv(BUNDLE/"demo_samples.csv")
    SPEAKERS=sorted(MANIFEST.speaker.astype(str).unique())
    WINDOW=float(CONFIG["speaker_identification_demo"]["window_seconds"])
except (FileNotFoundError,ValueError,KeyError) as error:
    st.title("How Similar Are Two Voices?")
    st.error(f"Paket export belum dapat dimuat: {error}")
    st.info("Letakkan isi export di folder Export di samping streamlit_app.py, lalu muat ulang halaman.")
    st.stop()

with st.sidebar:
    st.header("Voice similarity",icon=":material/graphic_eq:")
    st.caption("How Similar Are Two Voices?")

PAGES=[
    st.Page(overview,title="Ringkasan",icon=":material/dashboard:",default=True),
    st.Page(microphone_page,title="Mikrofon realtime",icon=":material/mic:",url_path="mikrofon"),
    st.Page(compare_page,title="Bandingkan suara",icon=":material/compare_arrows:",url_path="bandingkan"),
    st.Page(graph_page,title="Graph & Dijkstra",icon=":material/route:",url_path="graph"),
    st.Page(dataset_page,title="Eksplorasi dataset",icon=":material/dataset:",url_path="dataset"),
    st.Page(evaluation_page,title="Evaluasi & ketahanan",icon=":material/analytics:",url_path="evaluasi")]
selected_page=st.navigation({"Demo":PAGES[:4],"Eksplorasi":PAGES[4:]},expanded=True)
with st.sidebar:
    st.space("small")
    METHOD=st.selectbox("Representasi suara",core.METHODS,index=2,format_func=METHOD_LABELS.get,key="active_method")
    st.caption(f"{len(MANIFEST)} rekaman · {len(SPEAKERS)} speaker")
    st.caption(" · ".join(SPEAKERS))
    st.badge("Model lokal · CPU",icon=":material/computer:",color="green")
    with st.expander("Tentang eksperimen"):
        st.caption("MFCC + cosine, ECAPA pretrained, dan ECAPA fine-tuned. Tidak ada training ulang pada dashboard ini. Seluruh hasil dibaca dari export Colab Anda.")
selected_page.run()
