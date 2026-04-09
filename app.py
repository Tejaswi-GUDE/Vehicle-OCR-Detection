import ssl
import re
import cv2
import easyocr
import numpy as np
import streamlit as st
from PIL import Image

ssl._create_default_https_context = ssl._create_unverified_context
st.set_page_config(page_title="Vehicle Number Scanner", page_icon="🚗", layout="centered")

@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'], gpu=False)

reader = load_reader()

INDIAN_PLATE_PATTERNS = [
    r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$',
    r'^[A-Z]{2}[0-9]{2}[A-Z]{1}[0-9]{4}$',
    r'^[0-9]{2}BH[0-9]{4}[A-HJ-NP-Z]{1,2}$'
]

def clean_text(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())

def normalize_common_misreads(text):
    chars = list(text)

    if len(chars) >= 4:
        for i in range(len(chars) - 4, len(chars)):
            if chars[i] == 'O':
                chars[i] = '0'
            elif chars[i] == 'I':
                chars[i] = '1'
            elif chars[i] == 'Z':
                chars[i] = '2'
            elif chars[i] == 'S':
                chars[i] = '5'
            elif chars[i] == 'B':
                chars[i] = '8'

    return ''.join(chars)

def is_valid_indian_plate(text):
    return any(re.match(pattern, text) for pattern in INDIAN_PLATE_PATTERNS)

def plate_score(text, conf):
    score = conf
    if is_valid_indian_plate(text):
        score += 2.0
    if 8 <= len(text) <= 10:
        score += 0.5
    return score

def generate_variants(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)

    otsu = cv2.threshold(bfilter, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    adaptive = cv2.adaptiveThreshold(
        bfilter, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    enlarged = cv2.resize(bfilter, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    return [gray, otsu, adaptive, enlarged]

def extract_best_plate(img_array):
    variants = generate_variants(img_array)
    candidates = []

    for variant in variants:
        results = reader.readtext(
            variant,
            detail=1,
            paragraph=False,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            decoder='beamsearch',
            beamWidth=10,
            contrast_ths=0.1,
            adjust_contrast=0.5,
            text_threshold=0.6,
            low_text=0.3,
            link_threshold=0.4
        )

        for res in results:
            text = clean_text(res[1])
            conf = float(res[2])

            if len(text) < 6:
                continue

            candidates.append((text, conf))

            corrected = normalize_common_misreads(text)
            if corrected != text:
                candidates.append((corrected, conf - 0.05))

    if not candidates:
        return None

    candidates.sort(key=lambda x: plate_score(x[0], x[1]), reverse=True)

    for text, conf in candidates:
        if is_valid_indian_plate(text):
            return text

    return candidates[0][0]

st.title("🚗 Vehicle Number Scanner")
st.write("Upload a vehicle image or cropped number plate image.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🚀 Scan Vehicle Number"):
        with st.spinner("Scanning with EasyOCR..."):
            plate_no = extract_best_plate(img_array)

            if plate_no:
                st.success(f"Extracted Number: {plate_no}")

                if is_valid_indian_plate(plate_no):
                    st.info("Valid Indian number plate format ✅")
                else:
                    st.warning("Text detected, but format validation is weak ⚠️")
            else:
                st.error("No valid plate detected. Try a clearer or cropped image.")

st.markdown("---")
st.caption("Developed with Streamlit + EasyOCR")