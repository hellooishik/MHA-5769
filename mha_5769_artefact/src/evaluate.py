import torch
from .model import SimpleCNN
from .data_loader import get_data_loaders
from .config import MODEL_SAVE_PATH
import torchvision.transforms as transforms
import torch.nn.functional as F

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # We only need the test loader
    _, testloader, classes = get_data_loaders()

    # Load the best model
    model = SimpleCNN(num_classes=10).to(device)
    
    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        print(f"Successfully loaded the best model checkpoint from {MODEL_SAVE_PATH}.")
    except Exception as e:
        print(f"Failed to load model from {MODEL_SAVE_PATH}. Did you train it first?")
        return

    model.eval()
    
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    print("Running evaluation on test set...")
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print(f'\nOverall Accuracy on the 10000 test images: {100 * correct / total:.2f}%\n')
    
    for i in range(10):
        if class_total[i] > 0:
            print(f'Accuracy of {classes[i]:>5s} : {100 * class_correct[i] / class_total[i]:.2f}%')
            
    # Robustness testing
    print("\n--- Robustness Testing (Adding Gaussian Noise) ---")
    noise_level = 0.1
    robust_correct = 0
    robust_total = 0
    
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            
            # Add noise
            noise = torch.randn_like(inputs) * noise_level
            noisy_inputs = inputs + noise
            
            outputs = model(noisy_inputs)
            _, predicted = torch.max(outputs, 1)
            
            robust_total += labels.size(0)
            robust_correct += (predicted == labels).sum().item()
            
    print(f'Accuracy on noisy images (noise level={noise_level}): {100 * robust_correct / robust_total:.2f}%')

if __name__ == "__main__":
    evaluate_model()
