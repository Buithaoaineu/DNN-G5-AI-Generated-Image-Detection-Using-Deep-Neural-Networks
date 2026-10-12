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

- **Dataset name:** AI-Generated Image Detection Dataset
- **Dataset source:** [Kaggle Dataset](https://www.kaggle.com/datasets/97c52dbae208af8189b94d549ab0f2dfd90c3f5b14700d8506ba358d1a2d5c6f/data)
- **Task type:** Binary Image Classification
- **Classes:** Human-generated images and AI-generated images
- **Number of images:** 85,490 images
### Dataset layout

After downloading and extracting the dataset, the expected layout is:

```text
data/
├── train.csv
├── test.csv
├── train_data/
└── test_data_v2/
```

`train.csv` contains training metadata and labels. `test.csv` is used for the supplied test data and may not contain labels. The project also creates internal train, validation, and test splits from the labeled training data.

### Internal data split

The current experiment uses a balanced subset of 10,000 labeled images: 5,000 images per class, split as follows:

| Split | Images | Percentage |
|---|---:|---:|
| Training | 7,000 | 70% |
| Validation | 1,500 | 15% |
| Internal test | 1,500 | 15% |
| **Total** | **10,000** | **100%** |

These counts assume that at least 5,000 valid images are available for each class after cleaning and checking file paths. Update this section if the final dataset or split changes.

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

The complex model uses 4 residual blocks with shortcut connections, convolutional layers, batch normalization, global average pooling, and a binary classification output. The current implementation expects input images of shape `160 × 160 × 3`.

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
- **Deep Learning:** TensorFlow 
- **Data processing:** NumPy, Pandas
- **Image processing:** 
- **Visualization:** Matplotlib, Seaborn
- **Version control:** Git and GitHub
- **Development:** Jupyter Notebook, Visual Studio Code

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
.venv\Scripts\activate.ps1
```

### Install Dependencies

After the team has finalized the dependencies, run:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Dataset Setup

The dataset used in this project is available from the Kaggle competition:

[AI vs. Human-Generated Images](https://www.kaggle.com/datasets/97c52dbae208af8189b94d549ab0f2dfd90c3f5b14700d8506ba358d1a2d5c6f/data)

#### Steps to download the dataset

1. Log in to Kaggle and open the competition Data page.
2. Download the dataset files and extract them.
3. Place the files in the `data/` directory using the following structure:

```text
data/
├── train.csv
├── test.csv
├── train_data/
└── test_data_v2/
```

4. Keep the original image filenames so that they match the metadata.
5. Install the required Python packages using:

```bash
pip install -r requirements.txt
```

**Note:** The image folders are not included in this GitHub repository because of their size. Download the dataset from Kaggle before running the notebooks.

### Run prediction

Make sure the virtual environment is activated and the trained model
`src/models/complex_cnn.keras` and test data are available. From the repository root, run:

```bash
python -m src.predict

## 8. Team Members

| Member | Name | Responsibilities |
|---|---|---|
| Member 1 | Trần Phương Oanh | [Responsibilities] |
| Member 2 | Nguyễn Hà Vy | [Responsibilities] |
| Member 3 | Bùi Phương Thảo | [Responsibilities] |


## 9. Expected Deliverables

 Reproducible data preprocessing and train/validation/test split.
- Implementations of Simple CNN, Complex CNN, and a transfer-learning model.
- Model evaluation metrics and visualizations.
- A saved trained model and a prediction script.
- A final report comparing model performance and discussing limitations.

Update this list to reflect the models and experiments actually completed by the team.

## 10. Project Status


The project is under development. The data preparation and custom CNN workflow are being implemented. Model comparisons, transfer-learning experiments, final metrics, and conclusions should be added after the corresponding experiments have been completed.
