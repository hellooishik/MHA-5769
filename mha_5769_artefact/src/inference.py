import torch
import torchvision.transforms as transforms
from PIL import Image
from .model import SimpleCNN
from .config import MODEL_SAVE_PATH
import os

def predict_image(image_path):
    """
    Simulates real-world inference by taking an image path and predicting its class.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
               
    # Load model
    model = SimpleCNN(num_classes=10).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        model.eval()
    except Exception as e:
        print(f"Error loading model from {MODEL_SAVE_PATH}. Cannot perform inference. Did you train the model?")
        return None
        
    # Preprocess image
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image at {image_path}: {e}")
        return None
        
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # Add batch dimension
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        _, predicted_idx = torch.max(output, 1)
        
    predicted_class = classes[predicted_idx.item()]
    confidence = probabilities[predicted_idx.item()].item() * 100
    
    print(f"Predicted Class: {predicted_class} (Confidence: {confidence:.2f}%)")
    return predicted_class, confidence
