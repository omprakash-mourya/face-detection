# 🔐 Face Verification with Siamese Networks

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://face-detection-omprakash.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)

A secure, one-shot face verification system deployed on Streamlit Cloud. This application uses a custom **Siamese Neural Network** to verify identity with high accuracy even with minimal reference images.

### 🔴 [Live Demo: Click Here](https://face-detection-omprakash.streamlit.app/)

---

## 🧠 How It Works: The Siamese Architecture

Unlike traditional classification models that learn to classify "Person A" vs "Person B", this project uses **One-Shot Learning**. The model learns a **similarity function**, allowing it to verify faces it has never seen before during training.

### 1. Structure
The "Siamese" architecture consists of two identical Convolutional Neural Networks (CNNs) that share the **exact same weights**.

```mermaid
graph LR
    A[Input Image A] --> CNN[Shared CNN]
    B[Input Image B] --> CNN
    CNN --> E1[Embedding Vector A]
    CNN --> E2[Embedding Vector B]
    E1 --> L1[L1 Distance Layer]
    E2 --> L1
    L1 --> D[Dense + Sigmoid]
    D --> O[Similarity Score (0-1)]
```

### 2. The Process
1.  **Preprocessing**: 
    -   The system captures a **400x400** high-resolution crop from the live feed.
    -   It uses a strictly consistent pipeline to resize this to **100x100** using TensorFlow ops (`tf.image.resize`), ensuring zero discrepancy between training and inference data.
2.  **Embedding Generation**: 
    -   Both the live "Query" image and the stored "Reference" image are passed through the CNN.
    -   The CNN compresses the image into a dense 4096-dimensional vector (embedding).
3.  **Distance Calculation (`L1Dist`)**:
    -   A custom logic layer calculates the absolute difference between the two embeddings:  
        $$| Embedding_A - Embedding_B |$$
4.  **Verification**:
    -   The difference vector is passed through a final Dense layer with a Sigmoid activation.
    -   **Output**: A score between 0 and 1. (Score > 0.5 indicates a match).

---

## ✨ Key Features

*   **Real-time Stability**: Implements a custom resolution stabilization algorithm. The app waits for the camera feed to hit a stable **1280x720 (HD)** resolution before allowing verification, preventing low-quality captures.
*   **Session Isolation**: Uses ephemeral session storage. User data is isolated and automatically wiped when the browser tab is refreshed or closed.
*   **Precision Cropping**: Visual overlays guide the user to position their face perfectly within a 400x400 zone, which is then dynamically downscaled for the model wihout losing aspect ratio information.
*   **Strict Parity**: The preprocessing pipeline is engineered to mathematically match the original training environment (Keras v2/TensorFlow) to prevent "data drift" and ensure high accuracy.

---

## 🛠️ Installation

To run this locally:

1.  **Clone the repo**
    ```bash
    git clone https://github.com/omprakash-mourya/face-detection.git
    cd face-detection
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App**
    ```bash
    streamlit run app.py
    ```

---

## 📂 Project Structure

*   `app.py`: Main application logic (Streamlit + WebRTC).
*   `siamesemodelv2_1.h5`: The pre-trained Siamese Network model.
*   `requirements.txt`: Python dependencies.

---

<p align="center">
  Made with ❤️ by Omprakash Mourya
</p>
