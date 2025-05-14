# app.py (Versión para 5 Clases LOCALES)

import torch
from PIL import Image
from torchvision import transforms
import gradio as gr
import os # Para os.path.join

# Importa la definición de la clase de tu archivo architecture.py
from architecture import ResNetLungCancer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- 1. Configuración del Modelo (DEBE COINCIDIR CON EL ENTRENAMIENTO) ---
NUM_CLASSES = 5
# !!! USUARIO: ESTA LISTA DEBE COINCIDIR EXACTAMENTE (ORDEN Y NOMBRES) CON
# LA SALIDA DE `CLASS_NAMES` CUANDO EJECUTASTE architecture.py !!!
# Ejemplo basado en el orden alfabético de tus carpetas:
CLASS_NAMES = ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'notumor', 'squamous.cell.carcinoma']
# Verifica la salida de architecture.py para confirmar este orden.

model_name_from_training = f'lung_cancer_detection_model_{NUM_CLASSES}clases_local.pth'
model_path = os.path.join('Model', model_name_from_training)

# --- 2. Cargar Modelo ---
print(f"Loading model: {model_path} for {NUM_CLASSES} classes: {CLASS_NAMES}")
model = ResNetLungCancer(num_classes=NUM_CLASSES)
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
except FileNotFoundError:
    print(f"Error: Archivo del modelo no encontrado en {model_path}")
    print("Asegúrate de haber entrenado el modelo primero usando architecture.py y que el nombre/ruta sea correcto.")
    print("La aplicación Gradio no funcionará sin el modelo.")
    # Podrías lanzar una excepción aquí o manejarlo en la interfaz
    # Por ahora, la app se lanzará pero la predicción fallará si el modelo no carga.
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    # Idem

model = model.to(device)
model.eval() # ¡Muy importante para la predicción!
print("Model loaded successfully (o se intentó cargar).")


# --- 3. Definir Preprocesamiento (IDÉNTICO AL val_test_transform de architecture.py) ---
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 4. Función de Predicción para Gradio ---
def predict_lung_condition(input_image_pil):
    """
    Toma una imagen PIL de entrada, la preprocesa y devuelve un diccionario
    de etiquetas de clase y sus confianzas.
    """
    if input_image_pil is None:
        return {"Error": "No se proporcionó imagen."}
    
    try:
        # input_image ya es PIL si Gradio se configura bien, o numpy.
        # Si es numpy, convertir: image = Image.fromarray(input_image.astype('uint8'), 'RGB')
        # Si Gradio entrega PIL:
        image_rgb = input_image_pil.convert('RGB')

        input_tensor = preprocess(image_rgb)
        input_tensor = input_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
        
        # Obtener probabilidades con Softmax
        probabilities = torch.softmax(output, dim=1)[0] # Tomar la primera (y única) predicción del lote

        # Crear diccionario de {clase: probabilidad}
        confidences = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(NUM_CLASSES)}
        
        return confidences
    except AttributeError: # Si el modelo no se cargó
        return {"Error": "Modelo no cargado. Verifica la consola."}
    except Exception as e:
        print(f"Error durante la predicción en Gradio: {e}")
        return {"Error": str(e)}

# --- 5. Crear y Lanzar la Interfaz Gradio ---
# !!! USUARIO: AJUSTA LAS RUTAS DE EJEMPLO SI QUIERES QUE FUNCIONEN !!!
# Estas rutas deben apuntar a imágenes válidas en tu sistema donde ejecutas app.py
example_image_folder = os.path.join("image.png","image2.png","image3.jpeg","image4.jpeg") # Cambia "ruta/a/tu/dataset"
examples = []
if os.path.isdir(example_image_folder) and CLASS_NAMES:
    # Intenta tomar una imagen de cada clase como ejemplo
    for class_name_folder in CLASS_NAMES:
        class_folder_path = os.path.join(example_image_folder, class_name_folder)
        if os.path.isdir(class_folder_path):
            try:
                first_image = os.listdir(class_folder_path)[0]
                examples.append(os.path.join(class_folder_path, first_image))
            except IndexError:
                pass # Carpeta de clase vacía
    if not examples: # Si no se encontraron ejemplos
         examples = None # o una lista de rutas a imágenes que sepas que existen
else:
    examples = None


iface = gr.Interface(
    fn=predict_lung_condition,
    inputs=gr.Image(type="pil", label="Sube una imagen CT de pulmón"), # "pil" para PIL Image
    outputs=gr.Label(num_top_classes=NUM_CLASSES, label="Predicción"), # Muestra todas las clases y sus scores
    title="LungAI: Clasificador de Tejido Pulmonar (5 Clases)",
    description="Sube una imagen CT para clasificarla en una de las 5 categorías: Adenocarcinoma, Carcinoma de Células Grandes, Normal (Tumor Benigno), No Tumor (Sano), Carcinoma de Células Escamosas.",
    examples=examples, # Lista de rutas a imágenes de ejemplo
    allow_flagging='never'
)

if __name__ == "__main__":
    print("Lanzando interfaz de Gradio...")
    iface.launch(share=False) # share=True para un enlace público temporal (si es posible)