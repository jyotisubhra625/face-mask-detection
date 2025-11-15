# Face Mask Detection using CNN
![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-API-red)

## Overview

This project is a real-time face mask detection system that uses a Convolutional Neural Network (CNN) to identify whether a person is wearing a face mask. The model is built with Keras and TensorFlow and can be deployed as a web application using Streamlit.

## Dataset Used

The model was trained on a dataset of images containing people with and without face masks. The dataset is organized into two folders:
- `data/with_mask`
- `data/without_mask`

The dataset is not included in this repository to keep the size small. You can use your own dataset or find one online. A popular dataset for this task is the [Face Mask Detection dataset on Kaggle](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset).

## Model Architecture

The CNN model architecture is as follows:

- **Convolutional Layer 1:** 32 filters of size (3,3), ReLU activation
- **Max Pooling Layer 1:** (2,2) pool size
- **Convolutional Layer 2:** 64 filters of size (3,3), ReLU activation
- **Max Pooling Layer 2:** (2,2) pool size
- **Convolutional Layer 3:** 128 filters of size (3,3), ReLU activation
- **Max Pooling Layer 3:** (2,2) pool size
- **Flatten Layer**
- **Dense Layer:** 128 units, ReLU activation
- **Dropout Layer:** 0.5 dropout rate
- **Dense Layer (Output):** 1 unit, sigmoid activation

The model is compiled with the Adam optimizer and binary cross-entropy loss function.

## Training Steps

The model was trained for 5 epochs . The training process is detailed in the `Face Mask Detection using CNN.ipynb` notebook. The training history (accuracy and loss) is plotted to visualize the model's performance over time.

## Results

The model achieved an accuracy of over 95% on the validation set. The training and validation loss curves show that the model is not overfitting and generalizes well to new data.

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/jyotisubhra625/face-mask-detection.git
cd face-mask-detection
```

### 2. Create a virtual environment and install dependencies
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```
This will open the web application in your browser.

### 4. Run the Jupyter Notebook
To explore the data and the model training process, you can run the Jupyter Notebook:
```bash
jupyter notebook "Face Mask Detection using CNN.ipynb"
```
## 📥 Download Trained Model
Due to GitHub’s file size limits, the trained model is stored externally.

🔗 Download: https://drive.google.com/file/d/13_FabA_OzqzElkF8FLSbqr8mngp8XOwM/view?usp=sharing


## Project Structure
```
.
├── app.py
├── assets
│   ├── with_mask.png
│   └── without_mask.png
├── Face Mask Detection using CNN.ipynb
├── mask_detection_model.h5
├── requirements.txt
└── README.md
```

## Sample Predictions

Here are some sample predictions from the model:

**With Mask**

![With Mask](assets/with_mask.png)

**Without Mask**

![Without Mask](assets/without_mask.png)

## Future Improvements

- **Real-time video detection:** The current app only supports image uploads. The next step is to add real-time detection from a webcam feed.
- **Improve model accuracy:** The model can be improved by training on a larger and more diverse dataset.
- **Deploy to the cloud:** The Streamlit app can be deployed to a cloud platform like Heroku or Streamlit Sharing for public access.

## Requirements

The project requirements are listed in the `requirements.txt` file.

## Author

- **Subhrayoti**

Feel free to reach out with any questions or suggestions!
