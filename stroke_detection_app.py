import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import tempfile
import os

# For webcam streaming
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# =========================
# Model Definition (Flexible Flatten)
# =========================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.num_classes = num_classes
        self.fc1 = None  # build later
        self._initialized = False

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = torch.flatten(x, 1)

        if not self._initialized:
            self.fc1 = nn.Linear(x.size(1), self.num_classes).to(x.device)
            self._initialized = True

        x = self.fc1(x)
        return x


# =========================
# Helpers
# =========================
@st.cache_resource
def load_model(weights_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN()
    try:
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state, strict=False)
        loaded = True
    except Exception as e:
        loaded = False
        st.warning(f"⚠️ Could not load full weights: {e}")
    model.eval().to(device)
    return model, device, loaded

def preprocess_pil(img_pil, size=(224, 224)):
    t = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    return t(img_pil).unsqueeze(0)

def overlay_text(img_bgr, text, org=(10, 30), color=(0, 255, 0)):
    cv2.putText(img_bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return img_bgr

def play_beep():
    # Plays a short sine beep in the browser via WebAudio (no files needed)
    st.components.v1.html(
        """
        <script>
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const o = ctx.createOscillator();
        const g = ctx.createGain();
        o.type = "sine";
        o.frequency.value = 880;      // Hz
        o.connect(g); g.connect(ctx.destination);
        o.start();
        g.gain.setValueAtTime(0.15, ctx.currentTime);
        g.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.5);
        o.stop(ctx.currentTime + 0.5);
        </script>
        """,
        height=0
    )

def softmax_stroke_prob(logits, stroke_index=1):
    probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
    return float(probs[stroke_index]), probs


# =========================
# Page
# =========================
st.set_page_config(page_title="Stroke Detection App", layout="centered")
st.title("🧠 Stroke Detection App")
st.caption("For research/demo only. Not a medical device.")

# Model path & options
weights_path = st.text_input("Model weights path (.pth):", value="stroke_model_weights.pth")
model, device, loaded = load_model(weights_path)

# Global controls
colA, colB, colC = st.columns(3)
with colA:
    alert_threshold = st.slider("Alert threshold (P(stroke))", 0.5, 0.99, 0.90, 0.01)
with colB:
    debounce_frames = st.slider("Debounce frames (consecutive)", 1, 30, 8, 1)
with colC:
    show_overlay = st.checkbox("Show overlay text", value=True)

st.divider()

mode = st.radio("Choose input type:", ["Image", "Video", "Webcam (Live)"])

stroke_index = 1  # change if your class order is different: [no_stroke, stroke]


# =========================
# Image Prediction
# =========================
if mode == "Image":
    image_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

    if image_file and st.button("Predict Stroke"):
        image = Image.open(image_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        x = preprocess_pil(image).to(device)

        with torch.no_grad():
            outputs = model(x)
            p_stroke, probs = softmax_stroke_prob(outputs, stroke_index)

        pred = int(np.argmax(probs))
        label = "Stroke Detected!!" if pred == stroke_index else "No Stroke Detected"
        color = "🔴" if pred == stroke_index else "🟢"
        st.subheader(f"{color} {label}")
        st.write(f"**P(stroke):** {p_stroke:.2f}")

        if p_stroke >= alert_threshold:
            play_beep()


# =========================
# Video Prediction (file)
# =========================
if mode == "Video":
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if video_file and st.button("Analyze Video"):
        # Save temp video file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stframe = st.empty()
        progress = st.progress(0)

        consec = 0
        any_alert = False
        processed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # Process every 5th frame for efficiency
            if frame_count % 5 != 0:
                continue

            # Convert to RGB PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            x = preprocess_pil(pil_img).to(device)

            with torch.no_grad():
                outputs = model(x)
                p_stroke, _ = softmax_stroke_prob(outputs, stroke_index)

            processed += 1
            # Debounce
            if p_stroke >= alert_threshold:
                consec += 1
            else:
                consec = 0

            # Trigger alert when enough consecutive frames exceed threshold
            if consec >= debounce_frames and not any_alert:
                any_alert = True
                play_beep()

            # Overlay
            if show_overlay:
                color = (0, 0, 255) if p_stroke >= alert_threshold else (0, 200, 0)
                overlay_text(frame, f"P(stroke)={p_stroke:.2f}", (10, 30), color)
                if consec >= debounce_frames:
                    overlay_text(frame, "ALERT: STROKE DETECTED!", (10, 70), (0, 0, 255))

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            progress.progress(min(frame_count / max(1, total_frames), 1.0))

        cap.release()

        if processed == 0:
            st.warning("No frames were processed from the video.")
        else:
            st.success("✅ Analysis complete.")
            if any_alert:
                st.error("⚠️ Stroke event flagged in this clip (debounced).")
            else:
                st.info("No debounced stroke event detected.")


# =========================
# Webcam (Live) — browser webcam with alert
# =========================
if mode == "Webcam (Live)":
    if "consec" not in st.session_state:
        st.session_state.consec = 0
    if "trigger_alert" not in st.session_state:
        st.session_state.trigger_alert = False

    class LiveTransformer(VideoTransformerBase):
        def __init__(self):
            self.size = (224, 224)

        def transform(self, frame: av.VideoFrame):
            img = frame.to_ndarray(format="bgr24")
            # Convert to PIL for the same preprocessing
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            x = preprocess_pil(pil_img, size=self.size).to(device)

            with torch.no_grad():
                outputs = model(x)
                p_stroke, _ = softmax_stroke_prob(outputs, stroke_index)

            # Debounce logic
            if p_stroke >= alert_threshold:
                st.session_state.consec += 1
            else:
                st.session_state.consec = 0

            if st.session_state.consec >= debounce_frames:
                st.session_state.trigger_alert = True
                st.session_state.consec = 0  # reset after triggering

            # Overlay
            if show_overlay:
                color = (0, 0, 255) if p_stroke >= alert_threshold else (0, 200, 0)
                cv2.putText(img, f"P(stroke)={p_stroke:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                if st.session_state.trigger_alert:
                    cv2.putText(img, "ALERT!", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            return img

    webrtc_streamer(
        key="stroke-webcam",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": False},
        video_transformer_factory=LiveTransformer,
    )

    # Play a beep in the page when the alert flag is set
    if st.session_state.trigger_alert:
        play_beep()
        # Reset the alert flag so we don't beep continuously
        st.session_state.trigger_alert = False

    st.info("Tip: Use the gear icon on the webcam widget to choose your camera.")

# =========================
# Footer
# =========================
st.caption("⚠️ This demo is not a medical device. Validate thoroughly before any real-world use.")
