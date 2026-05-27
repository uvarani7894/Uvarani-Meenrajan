CA-MPCNN Sugarcane Disease Classification
Overview
This repository contains the PyTorch implementation of the proposed:
Cross-Attention Multiscale Parallel Convolutional Neural Network (CA-MPCNN)
for sugarcane leaf disease classification.
The model integrates:
Multiscale Parallel CNN
Dense Connections
Inception Blocks
Cross-Attention Mechanism
---
Dataset Structure
Arrange the dataset as follows:
dataset/
│
├── Healthy/
├── Mosaic/
├── RedRot/
├── Rust/
└── YellowLeaf/
Each folder should contain corresponding images.
---
Requirements
Install the required packages:
```bash
pip install torch torchvision scikit-learn
```
---
How to Run
Step 1: Update Dataset Path
Open the file:
```python
CA_MPCNN_Implementation.py
```
Update:
```python
dataset_path = "path_to_sugarcane_dataset"
```
Example:
```python
dataset_path = "./dataset"
```
---
Step 2: Run Training
Execute:
```bash
python CA_MPCNN_Implementation.py
```
---
Training Configuration
Parameter	Value
Image Size	224 × 224
Batch Size	32
Epochs	200
Optimizer	Adam
Learning Rate	0.0001
Loss Function	CrossEntropyLoss
Dropout	0.5
---
Output
The trained model will be saved as:
```bash
CA_MPCNN_sugarcane_model.pth
```
Evaluation metrics displayed:
Accuracy
Precision
Recall
F1-score
---
Hardware Used
NVIDIA RTX 3090 GPU
Intel Core i9 Processor
64 GB RAM
---
Citation
If you use this implementation, please cite the corresponding manuscript.
