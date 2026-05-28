import csv
import os

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

DATASET_PATH = "/Users/alanmendozaguerrero/Desktop/CimaLSM/fotos/MSLwords1"
OUTPUT_CSV = "/Users/alanmendozaguerrero/Desktop/CimaLSM/fotos/keypoints.csv"

options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="/Users/alanmendozaguerrero/Desktop/LSM-/hand_landmarker.task"
    ),
    num_hands=1,
)
hands = vision.HandLandmarker.create_from_options(options)

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)

    # esto lo que hara es leer todas las diapositivasa :)
    # 21 landmarks x 3 coordenadas (x, y, z) = 63 columnas + label
    header = [f"{i}_{axis}" for i in range(21) for axis in ["x", "y", "z"]]
    header.append("label")
    writer.writerow(header)

    clases = sorted(os.listdir(DATASET_PATH))
    for clase in clases:
        clase_path = os.path.join(DATASET_PATH, clase)
        if not os.path.isdir(clase_path):
            continue

        print(f"Procesando clase {clase}...")

        # Subcarpetas (individuos)
        for individuo in os.listdir(clase_path):
            individuo_path = os.path.join(clase_path, individuo)
            if not os.path.isdir(individuo_path):
                continue

            for img_file in os.listdir(individuo_path):
                if not img_file.endswith(".jpg"):
                    continue

                img_path = os.path.join(individuo_path, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                result = hands.detect(mp_image)

                if result.hand_landmarks:
                    landmarks = result.hand_landmarks[0]

                if result.hand_landmarks:
                    landmarks = result.hand_landmarks[0]
                    row = []
                    for lm in landmarks:
                        row.extend([lm.x, lm.y, lm.z])
                    row.append(int(clase))
                    writer.writerow(row)

hands.close()
print(f"Keypoints guardados en {OUTPUT_CSV}")
