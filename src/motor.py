import cv2
import joblib
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

# Rutas
MODEL_PATH = "/Users/alanmendozaguerrero/Desktop/CimaLSM/modelo.pkl"
LANDMARKER_PATH = "/Users/alanmendozaguerrero/Desktop/CimaLSM/hand_landmarker.task"

# Traducciones
traducciones = {
    1: "Buenos días",
    2: "Buenas tardes",
    3: "Buenas noches",
    4: "Gracias",
    5: "Por favor",
    6: "Hasta luego",
    7: "Adiós",
    46: "Abuela",
    47: "Abuelo",
    48: "Esposa",
    49: "Esposo",
    50: "Familia",
    51: "Mamá",
    52: "Papá",
    53: "Hermano",
    54: "Hermana",
    55: "Hijo",
    56: "Hija",
    57: "Tío",
    58: "Tía",
    59: "Primo",
    60: "Prima",
    61: "Bebé",
    62: "Niño",
    63: "Niña",
    64: "Joven",
    65: "Adulto",
    170: "Yo",
    171: "Tú",
    172: "Él",
    173: "Ella",
    174: "Nosotros",
    175: "Ustedes",
    176: "Ellos",
    177: "Ellas",
    178: "Mío",
    179: "Tuyo",
    250: "A",
    251: "B",
    252: "C",
    253: "D",
    254: "E",
    255: "F",
    256: "G",
    257: "H",
    258: "I",
    259: "J",
    260: "K",
    261: "L",
    262: "M",
    263: "N",
    264: "O",
    265: "P",
    266: "R",
    267: "S",
    268: "T",
    269: "U",
    270: "V",
    271: "W",
    272: "X",
    273: "Y",
    274: "Z",
}

# Cargar modelo
print("Cargando modelo...")
clf = joblib.load(MODEL_PATH)

# Configurar MediaPipe
options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=LANDMARKER_PATH), num_hands=1
)
hands = vision.HandLandmarker.create_from_options(options)


def normalizar(landmarks):
    puntos = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    muneca = puntos[0]
    puntos = puntos - muneca
    max_dist = np.max(np.abs(puntos)) + 1e-6
    puntos = puntos / max_dist
    return puntos.flatten()


# Cámara
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("No se puede abrir la cámara")
    exit()

print("Listo. Presiona Q para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = hands.detect(mp_image)

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]
        keypoints = normalizar(landmarks)
        label = clf.predict([keypoints])[0]
        texto = traducciones.get(label, str(label))

        # Dibujar landmarks
        h, w, _ = frame.shape
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        conexiones = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (0, 5),
            (5, 6),
            (6, 7),
            (7, 8),
            (0, 9),
            (9, 10),
            (10, 11),
            (11, 12),
            (0, 13),
            (13, 14),
            (14, 15),
            (15, 16),
            (0, 17),
            (17, 18),
            (18, 19),
            (19, 20),
            (5, 9),
            (9, 13),
            (13, 17),
        ]
        for start, end in conexiones:
            x1, y1 = int(landmarks[start].x * w), int(landmarks[start].y * h)
            x2, y2 = int(landmarks[end].x * w), int(landmarks[end].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # Mostrar texto
        cv2.putText(frame, texto, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv2.imshow("LSM - Reconocimiento de señas", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
