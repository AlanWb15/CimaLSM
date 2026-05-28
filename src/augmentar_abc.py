import csv
import os

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

ABC_PATH = "/Users/alanmendozaguerrero/Desktop/CimaLSM/ABCensenas"
OUTPUT_CSV = "/Users/alanmendozaguerrero/Desktop/CimaLSM/fotos/keypoints_abc.csv"
MODEL_PATH = "/Users/alanmendozaguerrero/Desktop/CimaLSM/hand_landmarker.task"

# Configurar MediaPipe
options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH), num_hands=1
)
hands = vision.HandLandmarker.create_from_options(options)


def normalizar(landmarks):
    puntos = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    muneca = puntos[0]
    puntos = puntos - muneca
    max_dist = np.max(np.abs(puntos)) + 1e-6
    puntos = puntos / max_dist
    return puntos.flatten()


def augmentar(img, n=100):
    variaciones = []
    h, w = img.shape[:2]
    for _ in range(n):
        aug = img.copy()
        # Brillo
        factor = np.random.uniform(0.6, 1.4)
        aug = np.clip(aug * factor, 0, 255).astype(np.uint8)
        # Rotación
        angulo = np.random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angulo, 1)
        aug = cv2.warpAffine(aug, M, (w, h))
        # Zoom
        scale = np.random.uniform(0.85, 1.15)
        M2 = cv2.getRotationMatrix2D((w // 2, h // 2), 0, scale)
        aug = cv2.warpAffine(aug, M2, (w, h))
        # Espejo horizontal aleatorio
        if np.random.random() > 0.5:
            aug = cv2.flip(aug, 1)
        variaciones.append(aug)
    return variaciones


# Mapeo letra → número de clase (continuando desde 250)
letras = sorted([f for f in os.listdir(ABC_PATH) if f.endswith(".jpg")])
label_map = {}
for i, archivo in enumerate(letras):
    letra = os.path.splitext(archivo)[0].upper()
    label_map[letra] = 250 + i

print(f"Letras encontradas: {list(label_map.keys())}")

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    header = [f"{i}_{axis}" for i in range(21) for axis in ["x", "y", "z"]]
    header.append("label")
    writer.writerow(header)

    for archivo in letras:
        letra = os.path.splitext(archivo)[0].upper()
        label = label_map[letra]
        img_path = os.path.join(ABC_PATH, archivo)
        img = cv2.imread(img_path)

        if img is None:
            print(f"No se pudo leer {archivo}")
            continue

        variaciones = augmentar(img, n=150)
        exitos = 0
        print(f"Procesando letra {letra}...")
        for var in variaciones:
            img_rgb = cv2.cvtColor(var, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            result = hands.detect(mp_image)

            if result.hand_landmarks:
                row = list(normalizar(result.hand_landmarks[0]))
                row.append(label)
                writer.writerow(row)
                exitos += 1

        print(f"✅ {letra}: {exitos} muestras generadas")

hands.close()
print(f"\n✅ Guardado en {OUTPUT_CSV}")
