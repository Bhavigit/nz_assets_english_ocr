"""
English OCR Pipeline with Arabic Competition Filtering

Pipeline:
1. Detect text regions using EasyOCR English model
2. Crop detected regions from original image
3. Upscale cropped region for better recognition
4. Recognize text using English and Arabic models
5. Apply filtering rules to remove Arabic/noise
6. Save results to CSV and annotated images

"""

import easyocr
import glob
import os
import cv2
import csv
import sys
import re
import numpy as np


def load_readers():
    """
    Initialize EasyOCR readers for English and Arabic.
    Returns:
        reader_en: English OCR reader
        reader_ar: Arabic OCR reader
    """
    print("Loading English Model...")
    reader_en = easyocr.Reader(
        ['en'],
        gpu=False,
        recog_network="english_g2",
        model_storage_directory=r"C:\Users\KarriBhavya\PycharmProjects\nz_assets_text_v1-arabic\nz_assets_text_v1-arabic\models",
        download_enabled=False
    )

    print("Loading Arabic Model...")
    reader_ar = easyocr.Reader(
        ['ar'],
        gpu=False,
        model_storage_directory=r"C:\Users\KarriBhavya\PycharmProjects\nz_assets_text_v1-arabic\nz_assets_text_v1-arabic\models",
        download_enabled=False
    )

    return reader_en, reader_ar


def preprocess_for_detection(image):
    """
    Apply contrast enhancement to improve text detection.
    Returns:
        detection-ready grayscale image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def upscale_crop(cropped):
    """
    Upscale cropped region to improve character recognition.
    """
    try:
        return cv2.resize(
            cropped,
            None,
            fx=2.0,
            fy=2.0,
            interpolation=cv2.INTER_CUBIC
        )
    except Exception:
        return cropped


def apply_filters(en_text, en_conf, ar_conf):
    """
    Apply filtering rules:
    - Remove empty text
    - Remove Arabic-dominant predictions
    - Remove non-English garbage
    - Remove single-symbol noise
    """
    if not en_text:
        return False

    if ar_conf > (en_conf + 0.2):
        return False

    if not re.search('[a-zA-Z]', en_text):
        return False

    if len(en_text) < 2 and not en_text.isalnum():
        return False

    return True


def process_images(folder_path):
    """
    Main processing pipeline.
    """
    image_files = sorted(
        [f for f in glob.glob(os.path.join(folder_path, "*"))
         if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    )

    reader_en, reader_ar = load_readers()

    output_directory = r"C:\Users\KarriBhavya\PycharmProjects\nz_assets_text_v1-arabic\nz_assets_text_v1-arabic\test_data\new1outputs"
    os.makedirs(output_directory, exist_ok=True)

    csv_path = os.path.join(output_directory, "ocr_results.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Image Name", "Extracted Text", "Confidence", "Points"])

        for image_file in image_files:
            print("Processing:", os.path.basename(image_file))

            image = cv2.imread(image_file)
            if image is None:
                continue

            detection_image = preprocess_for_detection(image)

            # Detect text regions
            detections = reader_en.readtext(
                detection_image,
                detail=1,
                decoder="beamsearch",
                mag_ratio=1.5,
                text_threshold=0.5,
                low_text=0.4
            )

            detected_count = 0

            for points, _, _ in detections:

                xmin = int(min(p[0] for p in points))
                xmax = int(max(p[0] for p in points))
                ymin = int(min(p[1] for p in points))
                ymax = int(max(p[1] for p in points))

                xmin, xmax = max(0, xmin), max(0, xmax)
                ymin, ymax = max(0, ymin), max(0, ymax)

                cropped = image[ymin:ymax, xmin:xmax]
                if cropped.size == 0:
                    continue

                cropped_upscaled = upscale_crop(cropped)

                # Recognition
                en_results = reader_en.readtext(cropped_upscaled, detail=1)
                en_text = " ".join([r[1] for r in en_results]).strip()
                en_conf = max([r[2] for r in en_results], default=0)

                ar_results = reader_ar.readtext(cropped_upscaled, detail=1)
                ar_conf = max([r[2] for r in ar_results], default=0)

                if not apply_filters(en_text, en_conf, ar_conf):
                    continue

                # Save results
                writer.writerow([
                    os.path.basename(image_file),
                    en_text,
                    en_conf,
                    points
                ])

                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                detected_count += 1

            if detected_count > 0:
                save_path = os.path.join(output_directory, os.path.basename(image_file))
                cv2.imwrite(save_path, image)
                print(f"  -> Saved: {detected_count} texts.")

    print("Done.")


if __name__ == "__main__":
    process_images(
        r"C:\Users\KarriBhavya\PycharmProjects\nz_assets_text_v1-arabic\nz_assets_text_v1-arabic\test_data\input_images"
    )
