import cv2
import pytesseract
import numpy as np
import re


class OCRReports:
    def __init__(self, path: str):
        self.path = path

    # -------------------------
    # LOAD IMAGE
    # -------------------------
    def load_image(self):
        img = cv2.imread(self.path)
        if img is None:
            raise ValueError(f" Image not found: {self.path}")
        return img

    # -------------------------
    # HIGH DETAIL PREPROCESS
    # -------------------------
    def preprocess(self, img, scale=3):

        #  Aggressive upscale
        img = cv2.resize(
            img,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #  Edge-preserving denoise
        gray = cv2.bilateralFilter(gray, 9, 75, 75)

        #  Sharpen (VERY IMPORTANT for digits)
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        sharp = cv2.filter2D(gray, -1, kernel)

        #  Otsu threshold (better than adaptive for clean images)
        _, thresh = cv2.threshold(
            sharp, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        return thresh

    # -------------------------
    # ROI (optional)
    # -------------------------
    def extract_roi(self, img):
        h, w = img.shape
        return img[int(h * 0.15):int(h * 0.95), :]

    # -------------------------
    # MULTI-SCALE OCR
    # -------------------------
    def extract_text_multiscale(self, img):

        scales = [2, 3, 4]   #  multiple zoom levels
        results = []

        for scale in scales:
            processed = self.preprocess(img, scale)
            roi = self.extract_roi(processed)

            config = r'''
            --oem 3
            --psm 6
            -c preserve_interword_spaces=1
            '''

            text = pytesseract.image_to_string(roi, config=config)
            results.append(text)

        #  Choose best (longest = most info)
        best_text = max(results, key=len)
        return best_text

    # -------------------------
    # CLEAN TEXT
    # -------------------------
    def clean_text(self, text):
        text = re.sub(r'[^\w\s./-]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    # -------------------------
    # RUN
    # -------------------------
    def run(self):
        img = self.load_image()

        #  Multi-scale OCR instead of single pass
        text = self.extract_text_multiscale(img)

        clean = self.clean_text(text)

        return clean