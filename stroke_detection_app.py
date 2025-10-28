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

# Model Definition
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 111 * 111, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 111 * 111)
        x = self.fc1(x)
        return x

# Load Model
model = SimpleCNN()
model.load_state_dict(torch.load("stroke_model_weights.pth", map_location=torch.device("cpu")))
model.eval()

# Transform for input frames/images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Streamlit Interface
st.title("Stroke Detection App")

st.write("Upload a **face image or a short video** to predict stroke likelihood based on mouth shape.")

option = st.radio("Choose input type:", ["Image", "Video"])

# Image Prediction
if option == "Image":
    image_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

    if image_file and st.button("Predict Stroke"):
        image = Image.open(image_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess
        image_tensor = transform(image).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1).numpy()[0]
            prediction = int(np.argmax(probs))
            confidence = probs[prediction]

        label = "⚠️ Stroke Detected" if prediction == 1 else "No Stroke"
        st.subheader(label)
        st.write(f"**Confidence:** {confidence*100:.2f}%")


# Video Prediction
if option == "Video":
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if video_file and st.button("Analyze Video"):
        # Save temp video file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        predictions = []
        frame_count = 0

        stframe = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            if frame_count % 5 != 0:
                continue

            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            image_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                outputs = model(image_tensor)
                probs = torch.softmax(outputs, dim=1).numpy()[0]
                prediction = int(np.argmax(probs))
                predictions.append(prediction)

            # Live preview
            stframe.image(frame_rgb, channels="RGB", use_column_width=True)

        cap.release()

        if predictions:
            stroke_ratio = np.mean(predictions)
            label = "⚠️ Stroke Detected" if stroke_ratio > 0.5 else "No Stroke"
            st.subheader(label)
            st.write(f"**Stroke likelihood:** {stroke_ratio*100:.2f}%")
        else:
            st.warning("No frames were processed from the video.")
