import argparse
import sys
import os

# Add src to Python path so we can import from it easily
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.train import train_model
from src.evaluate import evaluate_model
from src.inference import predict_image

def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 Image Classification AI System")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'evaluate', 'infer'],
                        help="Mode to run the system: train, evaluate, or infer")
    parser.add_argument('--image_path', type=str, default='',
                        help="Path to image for inference mode")
                        
    args = parser.parse_args()
    
    print(f"--- Running AI System Pipeline in {args.mode.upper()} mode ---")
    
    if args.mode == 'train':
        print("Initializing Training Pipeline...")
        train_model()
    elif args.mode == 'evaluate':
        print("Initializing Evaluation Pipeline...")
        evaluate_model()
    elif args.mode == 'infer':
        if not args.image_path:
            print("Error: --image_path is required for inference mode.")
            return
        print(f"Initializing Inference for image: {args.image_path}")
        predict_image(args.image_path)

if __name__ == "__main__":
    main()
