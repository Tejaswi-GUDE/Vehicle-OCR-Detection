import cv2
import easyocr
import re
import os
import numpy as np

reader = easyocr.Reader(['en'], gpu=False)

INDIAN_PLATE_PATTERNS = [
    r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$',      # MH12AB1234
    r'^[A-Z]{2}[0-9]{2}[A-Z]{1}[0-9]{4}$',        # KA08J9192
    r'^[0-9]{2}BH[0-9]{4}[A-HJ-NP-Z]{1,2}$'       # 22BH1234AB / 22BH1234A
]

def clean_text(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())

def normalize_common_misreads(text):
    replacements = {
        'O': '0',
        'I': '1',
        'Z': '2',
        'S': '5',
        'B': '8'
    }
    return ''.join(replacements.get(ch, ch) if ch.isalpha() and i >= len(text)-4 else ch
                   for i, ch in enumerate(text))

def is_valid_indian_plate(text):
    return any(re.match(pattern, text) for pattern in INDIAN_PLATE_PATTERNS)

def plate_score(text, conf):
    score = conf

    if is_valid_indian_plate(text):
        score += 2.0

    if 8 <= len(text) <= 10:
        score += 0.5

    return score

def generate_variants(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)

    otsu = cv2.threshold(bfilter, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    adaptive = cv2.adaptiveThreshold(
        bfilter, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    enlarged = cv2.resize(bfilter, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    return {
        "gray": gray,
        "otsu": otsu,
        "adaptive": adaptive,
        "enlarged": enlarged
    }

def extract_vehicle_number(image_path):
    if not os.path.exists(image_path):
        return "Error: Image file not found!"

    img = cv2.imread(image_path)
    if img is None:
        return "Error: Unable to read image!"

    variants = generate_variants(img)
    candidates = []

    for name, variant in variants.items():
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

            candidates.append((text, conf, name))

            corrected = normalize_common_misreads(text)
            if corrected != text:
                candidates.append((corrected, conf - 0.05, name + "_corrected"))

    if not candidates:
        return "No text detected"

    candidates = sorted(candidates, key=lambda x: plate_score(x[0], x[1]), reverse=True)

    for text, conf, source in candidates:
        if is_valid_indian_plate(text):
            return text

    return candidates[0][0]

if __name__ == "__main__":
    test_image = "test_plate.jpg"
    print(f"Processing: {test_image}...")
    result = extract_vehicle_number(test_image)
    print(f"Final Output: {result}")