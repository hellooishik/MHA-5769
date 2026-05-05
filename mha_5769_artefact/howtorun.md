# How to Run the AI Systems Application

This is a step-by-step guide on how to run your Image Classification artefact for the MHA 5769 Part 2 Assessment. 

## Step 1: Open Your Terminal
1. Open **Command Prompt** or **PowerShell** on your Windows computer.
2. Navigate to the project directory by typing the following command and pressing Enter:
   ```bash
   cd "C:\Users\oishi\Downloads\MHA 5769\mha_5769_artefact"
   ```

## Step 2: Install Dependencies
Before running the code, you need to ensure Python has all the required libraries (like PyTorch) installed. 
Run the following command to install them from the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## Step 3: Run the Training Pipeline
To train the Convolutional Neural Network (CNN) from scratch, run the main script in `train` mode. 
*Note: This will automatically download the MNIST dataset for you if it isn't already downloaded.*

```bash
python main.py --mode train
```
*You will see a progress bar for the dataset download, followed by the training loss and accuracy. The best model will be automatically saved to the `models/` folder.*

## Step 4: Run the Evaluation Pipeline
Once the model is trained, you need to evaluate its performance and test its robustness (as required by the assignment brief).

```bash
python main.py --mode evaluate
```
*This command loads your saved model, tests it against unseen images, prints the accuracy per class, and then performs a **Robustness Test** by injecting Gaussian noise into the images.*

## Step 5: Run Inference on a Single Image
To simulate a real-world deployment by predicting the class of a single image on your computer, use the `infer` mode with the included demo image:

```bash
python main.py --mode infer --image_path demo_image_7.png
```

---
**Video Demonstration Tip:** 
For your 30-minute demonstration video, simply record your screen while executing Steps 3, 4, and 5 sequentially. Explain what the system is doing at each step (training, validating robustness, and testing a real-world image).
