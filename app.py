"""
Driver Safety Monitor — Streamlit Dashboard (app.py)

Imports DriverSafetyPipeline from main.py (which in turn imports
process_frame() from each feature module).  Zero detection logic here.

Run:
    streamlit run app.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import streamlit as st
from datetime import datetime

from main import DriverSafetyPipeline
import config.config as config

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Driver Safety Monitor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Styles ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.block-container { padding-top: 2rem; }
.metric-card {
    background: #1e2130; border-radius: 10px;
    padding: 14px 18px; margin: 5px 0;
    border-left: 4px solid #444;
}
.metric-card.ok    { border-left-color: #00c853; }
.metric-card.warn  { border-left-color: #ff6d00; }
.metric-card.alert { border-left-color: #d50000; }
.metric-card .label { font-size: 0.7rem; color: #90a4ae;
                       text-transform: uppercase; letter-spacing: 1px; }
.metric-card .val   { font-size: 1.2rem; font-weight: 700; color: #eceff1; }
.metric-card .sub   { font-size: 0.68rem; color: #78909c; margin-top: 3px; }
.alert-banner {
    background: #b71c1c; color: white; border-radius: 8px;
    padding: 12px 18px; font-size: 1rem; font-weight: 700;
    text-align: center; margin-top: 8px; letter-spacing: 0.5px;
}
</style>
""", unsafe_allow_html=True)


# ─── Session state ────────────────────────────────────────────────────────────
for k, v in {"running": False, "pipeline": None, "cap": None,
             "alert_log": [], "last_results": {}}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🚗 Driver Safety Monitor")
    st.markdown("---")

    st.subheader("🔧 Enable Detectors")
    en_sleep = st.checkbox("💤 Drowsiness Detection",    value=True)
    en_dist  = st.checkbox("👁️ Distraction Detection",  value=True)
    en_yawn  = st.checkbox("😮 Yawning Detection",       value=True)
    en_drink = st.checkbox("🍺 Drink & Drive Detection", value=True)
    en_phone = st.checkbox("📱 Phone Detection",         value=True)

    st.markdown("---")
    st.subheader("⚙️ Settings")
    cam_idx = st.number_input("Camera Index", value=config.CAMERA_INDEX,
                              min_value=0, max_value=4, step=1)
    silent  = st.checkbox("🔇 Silent Mode (no alarm)", value=False)

    st.markdown("---")
    c1, c2 = st.columns(2)
    start_btn = c1.button("▶ Start", width="stretch", type="primary")
    stop_btn  = c2.button("⏹ Stop",  width="stretch")

    if st.button("🔄 Recalibrate Head Pose", width="stretch"):
        if st.session_state.pipeline:
            st.session_state.pipeline.recalibrate()
            st.success("Head pose recalibrated!")

    if st.button("↺ Reset All Counters", width="stretch"):
        if st.session_state.pipeline:
            st.session_state.pipeline.reset()
            st.success("Counters reset!")

    st.markdown("---")
    st.caption("Powered by MediaPipe · YOLOv8 · OpenCV")


# ─── Start / Stop ─────────────────────────────────────────────────────────────
if start_btn and not st.session_state.running:
    pl = DriverSafetyPipeline(silent=silent)
    pl.enable_sleep       = en_sleep
    pl.enable_distraction = en_dist
    pl.enable_yawning     = en_yawn
    pl.enable_drink       = en_drink
    pl.enable_phone       = en_phone

    cap = cv2.VideoCapture(int(cam_idx))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    st.session_state.pipeline = pl
    st.session_state.cap      = cap
    st.session_state.running  = True

if stop_btn and st.session_state.running:
    st.session_state.running = False
    if st.session_state.cap:
        st.session_state.cap.release()
        st.session_state.cap = None
    # NOTE: do NOT call pipeline.close() — that destroys shared MediaPipe instances
    st.session_state.pipeline = None


# ─── Layout ───────────────────────────────────────────────────────────────────
st.markdown("## 🚗 Driver Safety Monitor — Live Dashboard")
vid_col, info_col = st.columns([3, 2], gap="medium")
with vid_col:
    st.markdown("### 📹 Live Camera Feed")
    frame_ph = st.empty()
with info_col:
    st.markdown("### 📊 Detector Status")
    status_ph = st.empty()
alert_ph = st.empty()


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _card(label, value, sub, level):
    return (f'<div class="metric-card {level}"><div class="label">{label}</div>'
            f'<div class="val">{value}</div><div class="sub">{sub}</div></div>')


def _render(results):
    html = ""
    s = results.get("sleep")
    if s is not None:
        lvl = "alert" if s.get("drowsy") else ("warn" if s.get("counter", 0) > 10 else "ok")
        html += _card("💤 Drowsiness", "DROWSY ⚠" if s["drowsy"] else "Awake ✓",
                       f"EAR: {s.get('ear',0):.3f} | Closed frames: {s.get('counter',0)}", lvl)
    d = results.get("distraction")
    if d is not None:
        lvl = "alert" if d.get("distracted") else "ok"
        flags = ", ".join(filter(None, ["Gaze" if d.get("gaze") else "",
                                         "Head" if d.get("head") else "",
                                         "Off-centre" if d.get("center") else ""])) or "None"
        html += _card("👁️ Distraction", "DISTRACTED ⚠" if d["distracted"] else "Attentive ✓",
                       f"Y:{d.get('yaw',0):.0f}° P:{d.get('pitch',0):.0f}° | {flags}", lvl)
    yn = results.get("yawning")
    if yn is not None:
        lvl = "warn" if yn.get("yawning") else "ok"
        html += _card("😮 Yawning", "YAWNING ⚠" if yn["yawning"] else "Normal ✓",
                       f"MAR: {yn.get('mar',0):.3f} | Total yawns: {yn.get('yawn_count',0)}", lvl)
    dk = results.get("drink")
    if dk is not None:
        state = dk.get("state", "IDLE")
        lvl = {"IDLE":"ok","POSSIBLE_DRINKING":"warn","DRINKING":"warn","ALERT":"alert"}.get(state,"ok")
        html += _card("🍺 Drink & Drive", f"{state} {'🚨' if state=='ALERT' else ''}",
                       f"Risk: {dk.get('risk',0):.1f}/3.0 | Events: {dk.get('events',0)}", lvl)
    ph = results.get("phone")
    if ph is not None:
        state = ph.get("state", "IDLE")
        lvl = {"IDLE":"ok", "POSSIBLE_PHONE_USE":"warn", "CONFIRMED_PHONE_USE":"warn", "ALERT":"alert"}.get(state,"ok")
        html += _card("📱 Phone Detection", f"{state} {'🚨' if state=='ALERT' else ''}",
                       f"Risk: {ph.get('risk',0):.1f}/3.0 | Events: {ph.get('events',0)}", lvl)
    return html


def _alerts(results):
    out = []
    if (s := results.get("sleep"))       and s.get("drowsy"):     out.append("DROWSY")
    if (d := results.get("distraction")) and d.get("distracted"): out.append("DISTRACTED")
    if (y := results.get("yawning"))     and y.get("yawning"):    out.append("YAWNING")
    if (k := results.get("drink"))       and k.get("state") in ("DRINKING", "ALERT"): out.append("DRINK & DRIVE 🚨")
    if (p := results.get("phone"))       and p.get("state") == "ALERT": out.append("PHONE USE 📱")
    return out


# ─── Camera loop (while-loop for smooth feed — no st.rerun flickering) ────────
if st.session_state.running:
    cap = st.session_state.cap
    pl  = st.session_state.pipeline

    if cap and cap.isOpened() and pl:
        pl.enable_sleep       = en_sleep
        pl.enable_distraction = en_dist
        pl.enable_yawning     = en_yawn
        pl.enable_drink       = en_drink
        pl.enable_phone       = en_phone
        pl.silent             = silent

        # Continuous while-loop: updates placeholders in-place without full page rerun.
        # Clicking any Streamlit widget (Stop, checkbox, etc.) triggers a rerun that
        # breaks out of this loop naturally.
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame     = cv2.flip(frame, 1)
            results   = pl.process(frame)
            annotated = pl.annotate(frame, results)
            st.session_state.last_results = results

            # Update placeholders in-place (no page rebuild = no flicker)
            frame_ph.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                           width="stretch")
            status_ph.markdown(_render(results), unsafe_allow_html=True)

            active = _alerts(results)
            if active:
                alert_ph.markdown(
                    f'<div class="alert-banner">⚠️  ALERT: {" &nbsp;|&nbsp; ".join(active)}</div>',
                    unsafe_allow_html=True)
                for a in active:
                    entry = (datetime.now().strftime("%H:%M:%S"), a)
                    lg = st.session_state.alert_log
                    if not lg or lg[-1][1] != a:
                        lg.append(entry)
                        if len(lg) > 100: lg.pop(0)
            else:
                alert_ph.empty()

            time.sleep(0.03)  # ~30 fps cap

    else:
        st.session_state.running = False
        frame_ph.error("❌ Camera not available.")

else:
    if not st.session_state.last_results:
        frame_ph.markdown(
            '<div style="background:#1e2130;border-radius:12px;padding:80px 20px;'
            'text-align:center;color:#546e7a;font-size:1.15rem;">'
            '▶ Press <b>Start</b> in the sidebar to begin monitoring</div>',
            unsafe_allow_html=True)
    else:
        status_ph.markdown(_render(st.session_state.last_results), unsafe_allow_html=True)


# ─── Alert log ────────────────────────────────────────────────────────────────
log = st.session_state.alert_log
if log:
    st.markdown("---")
    st.markdown("### 📋 Alert History (last 20)")
    rows = "".join(
        f"<tr><td style='padding:5px 12px;color:#78909c'>{t}</td>"
        f"<td style='padding:5px 12px;color:#ef5350;font-weight:600'>{a}</td></tr>"
        for t, a in reversed(log[-20:])
    )
    st.markdown(
        f"<table style='width:100%;border-collapse:collapse;background:#1e2130;"
        f"border-radius:8px;overflow:hidden'>"
        f"<thead><tr><th style='padding:6px 12px;color:#546e7a;text-align:left'>Time</th>"
        f"<th style='padding:6px 12px;color:#546e7a;text-align:left'>Alert</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>", unsafe_allow_html=True)
