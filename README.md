# AI-Generated Image Detection Using Deep Neural Networks

## 1. Project Overview

This project aims to investigate the use of Deep Neural Networks (DNNs), particularly Convolutional Neural Networks (CNNs), for image-based classification or segmentation.

The project focuses on preprocessing image data, designing CNN architectures, applying transfer learning and fine-tuning, and evaluating model performance.

## 2. Project Objectives

The main objectives are:

-
-
-

## 3. Dataset

- **Dataset name:** [To be determined]
- **Dataset source:** [Insert dataset URL]
- **Task type:** Image Classification
- **Classes:** [Insert class names]
- **Number of images:** [Insert dataset size]

The dataset will be divided into training, validation, and test sets. Data preprocessing will be applied to prepare the images for model training and evaluation.

> Note: The dataset details and task configuration will be updated after the dataset selection is finalized.

## 4. Methodology

### 4.1 Data Preprocessing

The preprocessing stage includes:

- Cleaning the dataset and handling invalid or corrupted images.
- Resizing images to a consistent input size.
- Normalizing pixel values.
- Applying data augmentation to improve generalization.
- Splitting the dataset into training, validation, and test sets.

### 4.2 Simple CNN

A custom CNN model will be developed using convolutional layers, pooling layers, and fully connected layers.

This model will serve as a baseline for evaluating the selected image task.

### 4.3 Complex CNN

A more complex CNN architecture will be designed using multiple CNN blocks, convolutional layers, pooling layers, and fully connected layers.

Its performance will be compared with the simple CNN model.

### 4.4 Transfer Learning and Fine-Tuning

A pretrained CNN model will be adapted to the selected task.

The experiments will investigate transfer learning and fine-tuning to evaluate whether pretrained features improve performance.

The pretrained architecture will be specified after the model selection process.

### 4.5 Model Evaluation

The models will be evaluated using appropriate performance metrics based on the selected task.

For image classification, potential metrics include accuracy, precision, recall, F1-score, and a confusion matrix.

For image segmentation, suitable metrics may include Intersection over Union (IoU) and Dice coefficient.

The final evaluation will compare the models' performance and discuss their strengths and limitations.

## 5. Repository Structure

```text
DNN-G5-AI-Generated-Image-Detection-Using-Deep-Neural-Networks/
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── docs/
│   ├── requirements.md
│   ├── dataset.md
│   └── report.md
├── notebooks/
├── src/
│   ├── data/
│   │   ├── preprocessing.py
│   │   └── dataset.py
│   ├── models/
│   │   ├── simple_cnn.py
│   │   ├── complex_cnn.py
│   │   └── transfer_learning.py
│   └── evaluation/
│       ├── evaluate.py
│       └── visualize.py
├── results/
│   ├── figures/
│   └── metrics/
├── tests/
├── README.md
├── requirements.txt
└── .gitignore
```

The structure may be adjusted as the project develops.

## 6. Technologies

The project is expected to use the following tools and technologies:

- **Programming language:** Python
- **Deep Learning:** TensorFlow or PyTorch
- **Data processing:** NumPy, Pandas
- **Image processing:** 
- **Visualization:** Matplotlib, Seaborn
- **Version control:** Git and GitHub

The final technology stack will be confirmed by the team.

## 7. Installation and Setup

### Prerequisites

- Python 3.10 or a compatible version supported by the selected deep learning framework.
- Git.
- A code editor such as Visual Studio Code.

### Clone the Repository

```bash
git clone https://github.com/Buithaoaineu/DNN-G5-AI-Generated-Image-Detection-Using-Deep-Neural-Networks.git
cd DNN-G5-AI-Generated-Image-Detection-Using-Deep-Neural-Networks
```

### Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### Install Dependencies

After the team has finalized the dependencies, run:

```bash
pip install -r requirements.txt
```

## 8. Team Members

| Member | Name | Responsibilities |
|---|---|---|
| Member 1 | Trần Phương Oanh | [Responsibilities] |
| Member 2 | Nguyễn Hà Vy | [Responsibilities] |
| Member 3 | Bùi Phương Thảo | [Responsibilities] |


## 9. Expected Deliverables

-
-
-

## 10. Project Status

The project is currently in the setup and planning stage. Dataset details, model implementations, experimental results, and final conclusions will be added as development progresses.
