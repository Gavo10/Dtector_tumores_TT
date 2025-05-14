# architecture.py (Versión para 5 Clases LOCALES con ImageFolder)

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models, datasets # datasets de torchvision para ImageFolder
from torch.utils.data import DataLoader
import os
import copy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import resnet50, ResNet50_Weights
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# --- Definición de Transformaciones (PIL -> Tensor) ---
# Estas se pasarán directamente a ImageFolder
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Considera stats del dataset
])

val_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Considera stats del dataset
])

# --- Definición del Modelo ResNetLungCancer ---
class ResNetLungCancer(nn.Module):
    def __init__(self, num_classes, use_pretrained=True):
        super(ResNetLungCancer, self).__init__()
        if use_pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V1
        else:
            weights = None
        self.resnet = resnet50(weights=weights)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.resnet(x)
        return self.fc(x)

# --- Funciones de Entrenamiento y Evaluación ---
def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs=50, device='cuda'):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    dataset_sizes = {'train': len(train_loader.dataset), 'valid': len(valid_loader.dataset)}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = valid_loader
            
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloader: # ImageFolder devuelve (inputs, labels)
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'valid':
                scheduler.step(epoch_acc)
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Learning rate: {current_lr}')
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
        print()
        
    print(f'Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model

def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    running_corrects = 0
    dataset_size = len(test_loader.dataset)
    
    with torch.no_grad():
        for inputs, labels in test_loader: # ImageFolder devuelve (inputs, labels)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            
    test_acc = running_corrects.double() / dataset_size
    print(f'Test Acc: {test_acc:.4f}')


# --- Bloque Principal de Ejecución ---
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Ruta al Dataset Local ---
    # !!! USUARIO: CAMBIA ESTA RUTA A TU CARPETA "dataset" !!!
    # Esta carpeta "dataset" debe contener las subcarpetas "train", "validation" y "test".
    # Y dentro de cada una de ellas, las 5 subcarpetas con los nombres de tus clases.
    data_dir = "C:/Users/obedc/Desktop/Datasetbueno2"
    # Ejemplo: data_dir = "D:/Proyecto_LungAI/dataset" 
    print(f"Loading local dataset from: {data_dir}")

    if not os.path.isdir(data_dir):
        raise ValueError(f"El directorio del dataset no existe: {data_dir}. Por favor, verifica la ruta.")

    # --- 2. Cargar Datasets Locales con ImageFolder ---
    try:
        train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
        valid_dataset = datasets.ImageFolder(os.path.join(data_dir, 'validation'), transform=val_test_transform)
        test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=val_test_transform)
    except FileNotFoundError as e:
        print(f"Error al cargar el dataset con ImageFolder: {e}")
        print("Asegúrate que la ruta principal 'data_dir' es correcta y que contiene las subcarpetas 'train', 'validation' y 'test'.")
        print("Dentro de 'train', 'validation' y 'test', deben estar las carpetas de tus 5 clases.")
        exit()
    except Exception as e:
        print(f"Ocurrió un error inesperado al cargar los datasets: {e}")
        exit()

    # --- 3. Obtener Nombres de Clases y Número de Clases ---
    NUM_CLASSES = len(train_dataset.classes)
    CLASS_NAMES = train_dataset.classes 
    
    print(f"Detectadas {NUM_CLASSES} clases: {CLASS_NAMES}")
    if NUM_CLASSES != 5: # Verificación
        print(f"ADVERTENCIA: Se detectaron {NUM_CLASSES} clases, pero se esperaban 5. Verifica la estructura de subcarpetas en '{os.path.join(data_dir, 'train')}' (y 'validation', 'test').")
        print("Los nombres de las subcarpetas de clase deben ser consistentes en train, validation y test.")
    
    # --- 4. Crear DataLoaders ---
    batch_size = 32
    num_workers = 4 if os.name == 'posix' else 0 # num_workers > 0 puede dar problemas en Windows directamente

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print("DataLoaders creados.")

    print(f"Number of training images: {len(train_dataset)}")
    print(f"Number of validation images: {len(valid_dataset)}")
    print(f"Number of test images: {len(test_dataset)}")

    # --- 5. Inicializar Modelo, Pérdida, Optimizador ---
    model = ResNetLungCancer(num_classes=NUM_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss() # Considera pesos si las clases están desbalanceadas

    pretrained_params = list(model.resnet.parameters())
    new_params = list(model.fc.parameters())
    optimizer = optim.Adam([
        {'params': pretrained_params, 'lr': 1e-5},
        {'params': new_params, 'lr': 1e-4}
    ], weight_decay=1e-6)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=7, verbose=True)

    # --- 6. Entrenar y Evaluar ---
    print(f"Starting training for {NUM_CLASSES} local classes: {CLASS_NAMES}...")
    num_epochs_to_train = 50 # Puedes ajustar esto
    trained_model = train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler,
                                num_epochs=num_epochs_to_train, device=device)
    print("Training finished.")
    print("Evaluating model on test set...")
    evaluate_model(trained_model, test_loader, device=device)

    # --- 7. Guardar Modelo ---
    model_dir = 'Model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    model_save_name = f'lung_cancer_detection_model_{NUM_CLASSES}clases_local.pth'
    onnx_save_name = f'lung_cancer_detection_model_{NUM_CLASSES}clases_local.onnx'
    model_save_path = os.path.join(model_dir, model_save_name)
    onnx_save_path = os.path.join(model_dir, onnx_save_name)

    torch.save(trained_model.state_dict(), model_save_path)
    print(f"Model weights saved to {model_save_path}")

    # --- 8. Exportar a ONNX (Opcional) ---
    try:
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        import onnx
        import onnxruntime # No es estrictamente necesario para exportar, pero sí para correr ONNX.
        torch.onnx.export(trained_model, dummy_input, onnx_save_path,
                          input_names=['input'], output_names=['output'],
                          opset_version=11, dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
        print(f"Model exported to ONNX format at {onnx_save_path}")
    except ImportError:
        print("ONNX libraries not installed (onnx, onnxruntime). Skipping ONNX export. Run: pip install onnx onnxruntime")
    except Exception as e:
        print(f"Failed to export model to ONNX: {e}")

    print("Script completed.")