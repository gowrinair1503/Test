import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from .utils import analyze_test_result
from PIL import Image

# Load model
model = YOLO("best.pt")

st.title("Home Test Kit Analyzer 🧪")
uploaded_file = st.file_uploader("Upload a test kit image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Run detection
    results = model(image, conf=0.25)
    res = results[0]

    test_type = "Unknown"
    for box in res.boxes:
        cls_id = int(box.cls[0])
        label = res.names[cls_id]
        conf = float(box.conf[0])

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if label == "D":
            test_crop = image[y1:y2, x1:x2]
            cv2.imwrite("test_crop.jpg", test_crop)
            st.image(test_crop, caption="Cropped Test Area", channels="BGR")
        else:
            test_type = label
            st.markdown(f"### 🧾 Detected Test Type: `{label}` (Confidence: {conf:.2f})")

    if test_type != "Unknown":
        result = analyze_test_result("test_crop.jpg", test_type)
        st.success(f"📋 **Result:** {result}")
    else:
        st.warning("Could not determine test type.")
