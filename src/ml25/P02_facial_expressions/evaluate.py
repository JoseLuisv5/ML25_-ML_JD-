import torch
from ml25.P02_facial_expressions.dataset import get_loader
from ml25.P02_facial_expressions.network import Network

NUM_CLASSES = 7  # Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

EMOTIONS_MAP = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}


def evaluate(split="val", model_name="modelo_1.pt", batch_size=256):
    """
    Evalúa el modelo en el split dado (val o test) y reporta:
    - Accuracy global
    - Matriz de confusión
    """
    # 1) Cargar datos
    dataset, loader = get_loader(split, batch_size=batch_size, shuffle=False)

    # 2) Cargar modelo entrenado
    modelo = Network(input_dim=48, n_classes=NUM_CLASSES)
    modelo.load_model(model_name)
    modelo.eval()
    device = modelo.device

    correct = 0
    total = 0
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)  # [real, pred]

    with torch.inference_mode():
        for batch in loader:
            imgs = batch["transformed"].to(device)
            labels = batch["label"].to(device)

            logits, _ = modelo(imgs)
            preds = torch.argmax(logits, dim=1)

            # accuracy global
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # matriz de confusión
            for t, p in zip(labels.view(-1), preds.view(-1)):
                cm[t.long(), p.long()] += 1

    acc = correct / total

    print(f"Evaluación en split = {split}")
    print(f"Accuracy global: {acc*100:.2f}%")
    print("Matriz de confusión (filas = etiqueta real, columnas = predicción):")
    print(cm)

    # Opcional: accuracy por clase
    print("\nAccuracy por clase:")
    for i in range(NUM_CLASSES):
        true_count = cm[i].sum().item()
        correct_i = cm[i, i].item()
        if true_count > 0:
            acc_i = correct_i / true_count * 100
            print(f"  {i} ({EMOTIONS_MAP[i]}): {acc_i:.2f}%")
        else:
            print(f"  {i} ({EMOTIONS_MAP[i]}): sin ejemplos en este split")

    return acc, cm


if __name__ == "__main__":
    evaluate(split="val")  # puedes cambiar a "test" si quieres evaluar en test
