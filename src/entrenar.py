import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

CSV_PATH = "/Users/alanmendozaguerrero/Desktop/CimaLSM/fotos/keypoints_combined.csv"
MODEL_PATH = "/Users/alanmendozaguerrero/Desktop/CimaLSM/modelo.pkl"
CLASSES_PATH = "/Users/alanmendozaguerrero/Desktop/CimaLSM/fotos/MSLwords1/classes.xlsx"

# Clases que queremos usar
# Greetings (1-7), Family (46-65), Pronouns (170-179)
CLASES_FILTRO = list(range(1, 8)) + list(range(46, 66)) + list(range(170, 180))

df = pd.read_csv(CSV_PATH)
df_clases = pd.read_excel(CLASSES_PATH)

# Filtrar solo las clases seleccionadas
df = df[df["label"].isin(CLASES_FILTRO)]

print(f"Total muestras después de filtrar: {len(df)}")
print(f"Total clases: {df['label'].nunique()}")

X_raw = df.drop("label", axis=1).values
y = df["label"].values


# Normalizar relativos a la muñeca
def normalizar(row):
    puntos = row.reshape(21, 3)
    muneca = puntos[0]
    puntos = puntos - muneca
    max_dist = np.max(np.abs(puntos)) + 1e-6
    puntos = puntos / max_dist
    return puntos.flatten()


X = np.array([normalizar(row) for row in X_raw])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Entrenando con {len(X_train)} muestras...")

clf = RandomForestClassifier(n_estimators=325, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Precisión: {acc * 100:.2f}%")

# Guardar modelo y mapeo de clases
joblib.dump(clf, MODEL_PATH)


print(f"✅ Modelo guardado en {MODEL_PATH}")
