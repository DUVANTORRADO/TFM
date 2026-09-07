import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from keras.models import load_model
from constants import NUM_EPOCHS, BATCH_SIZE, DATA_DIR, OUTPUTS_DIR, MODELS_DIR
from modelo import get_model as get_transformer
from modelo_CNN import get_model as get_cnn
from modelo_LSTM import get_model as get_lstm

# === CARGA DE DATOS (ajusta según tu pipeline) ===
def cargar_datos():
    """
    Carga los datos preprocesados desde data/keypoints/
    Debe devolver X_train, X_test, y_train, y_test en formato one-hot.
    """
    # Ejemplo: si tienes archivos .npy
    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
    y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
    return X_train, X_test, y_train, y_test

# === ENTRENAMIENTO Y EVALUACIÓN ===
def entrenar_y_evaluar(model_fn, nombre, X_train, y_train, X_test, y_test):
    print(f"\n--- Entrenando {nombre} ---")
    model = model_fn(y_train.shape[1])
    
    history = model.fit(
        X_train, y_train,
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.15,
        verbose=1
    )
    
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Guardar modelo
    model.save(os.path.join(MODELS_DIR, f"model_{nombre.lower()}_final.keras"))
    
    return {
        "name": nombre,
        "accuracy": acc,
        "loss": loss,
        "history": history.history,
        "y_true": y_true,
        "y_pred": y_pred_classes
    }

# === GENERAR GRÁFICAS ===
def generar_graficas(resultados):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for res in resultados:
        axes[0].plot(res["history"]["accuracy"], label=f"{res['name']} (acc)")
        axes[1].plot(res["history"]["loss"], label=f"{res['name']} (loss)")
    
    axes[0].set_title("Precisión durante entrenamiento")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Precisión")
    axes[0].legend()
    
    axes[1].set_title("Pérdida durante entrenamiento")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Pérdida")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "comparativa_curvas.png"), dpi=300)
    plt.close()

def generar_matrices_confusion(resultados):
    for res in resultados:
        cm = confusion_matrix(res["y_true"], res["y_pred"])
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Matriz de Confusión - {res['name']}")
        plt.xlabel("Predicción")
        plt.ylabel("Real")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTS_DIR, f"matriz_confusion_{res['name'].lower()}.png"), dpi=300)
        plt.close()

def mostrar_tabla_comparativa(resultados):
    print("\n" + "="*60)
    print("COMPARATIVA DE RENDIMIENTO")
    print("="*60)
    print(f"{'Modelo':<15} {'Exactitud':<15} {'Pérdida':<15}")
    print("-"*45)
    for res in resultados:
        print(f"{res['name']:<15} {res['accuracy']*100:.2f}%{'':<8} {res['loss']:.4f}")
    print("="*60)

# === MAIN ===
if __name__ == "__main__":
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    X_train, X_test, y_train, y_test = cargar_datos()
    
    modelos = [
        (get_cnn, "CNN"),
        (get_lstm, "LSTM"),
        (get_transformer, "Transformer")
    ]
    
    resultados = []
    for model_fn, nombre in modelos:
        res = entrenar_y_evaluar(model_fn, nombre, X_train, y_train, X_test, y_test)
        resultados.append(res)
    
    generar_graficas(resultados)
    generar_matrices_confusion(resultados)
    mostrar_tabla_comparativa(resultados)
    
    print("\n✅ Resultados guardados en carpeta 'outputs/'")