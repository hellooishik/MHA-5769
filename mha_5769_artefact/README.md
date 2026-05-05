# AI Systems Engineering (Part 2) Artefact: CIFAR-10 Image Classification

This repository contains the working artefact for the Part 2 assessment. It implements a complete AI engineering pipeline for Image Classification using PyTorch and the CIFAR-10 dataset.

## System Components

1. **Data Engineering:** Automated downloading, preprocessing, and augmentation of CIFAR-10 data (`src/data_loader.py`).
2. **Model Development:** Custom Convolutional Neural Network (CNN) defined in `src/model.py` and trained in `src/train.py`.
3. **Evaluation & Robustness:** Standard performance metrics and robustness testing against noisy data (`src/evaluate.py`).
4. **Inference Pipeline:** Simulation of real-world deployment for predicting unseen images (`src/inference.py`).

## Instructions for Execution

### 1. Environment Setup

Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment.

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Pipeline

The system is controlled via a single entry point script `main.py`.

#### Training the Model
To train the model from scratch (this will automatically download the dataset if it doesn't exist):
```bash
python main.py --mode train
```

#### Evaluating the Model
To evaluate the best saved model on the test set and perform robustness testing:
```bash
python main.py --mode evaluate
```

#### Running Inference
To simulate real-world usage by classifying a single image:
```bash
python main.py --mode infer --image_path path/to/your/image.jpg
```

## System Engineering Considerations

- **Reproducibility:** A `requirements.txt` file is provided to match the development environment.
- **Modularity:** The codebase is split into distinct logical components (data, model, training, evaluation) unlike a monolithic Jupyter Notebook.
- **Robustness:** Includes noise-injection during evaluation to test model resilience.
