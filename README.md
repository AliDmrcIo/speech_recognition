# Speech Recognition & Speaker Diarization (End-to-End AI Project)

🔗 **Live on:** [http://whosayswhat.duckdns.org](http://whosayswhat.duckdns.org)
This is the live production version of the project. You can directly upload your audio files and test the AI-based speech recognition and speaker diarization system from this link.

---

## Project Description

This project is a full-stack, production-ready **AI-powered speech analysis system**. Users can upload audio files, transcribe speech to text using state-of-the-art **ASR (Automatic Speech Recognition) models**, and automatically identify distinct **Speaker Diarization**. The system features a dual-mode processing engine (Fast vs. Pro) and exports color-coded transcripts in Word format.

The project is designed as a complete **End-to-End AI application**, covering:

*   Model serving & optimization
*   Audio signal processing
*   Speaker clustering algorithms
*   Containerization
*   Cloud deployment on limited resources (CPU-only)

---

## Project Goal

The main goal of this project is to build a **robust, scalable, and efficient AI application** that converts spoken language into structured, speaker-labeled text. The focus is not only on machine learning performance but also on:

*   **Resource Optimization:** Running heavy AI models on CPU-only infrastructure.
*   **User Experience:** Providing distinct modes for speed vs. accuracy.
*   **Real-world Utility:** Generating usable, formatted reports (.docx).
*   **Production Deployment:** Dockerized environment on AWS EC2.

---

## 🛠️ Technologies Used

### AI & Audio Processing

*   **Task:** Automatic Speech Recognition (ASR) & Speaker Diarization
*   **Models:**
    *   `WhisperX` (High-precision alignment & diarization)
    *   `Faster-Whisper` (Optimized inference speed)
    *   `Pyannote.audio` (Speaker segmentation & embedding)
*   **Algorithms:** `K-Means Clustering` (For custom speaker separation in Fast Mode)
*   **Libraries:** `PyTorch`, `Torchaudio`, `Librosa`, `Scikit-learn`

### Application Logic & Backend

*   **Language:** `Python 3.13`
*   **Logic:** Custom pipelines for audio chunking, feature extraction (MFCC), and transcript alignment.
*   **Export:** `python-docx` (For generating formatted Word documents)
*   **Environment:** `Python-Dotenv` (Configuration management)

### Frontend (User Interface)

*   **Framework:** `Streamlit`
*   **Design:** Custom layout with distinct processing modes (Fast/Pro)
*   **Interactivity:** Real-time audio playback and file handling

### DevOps, Container & Cloud

*   **Containerization:** `Docker` (Optimized Slim Image)
*   **Cloud Provider:** `AWS EC2` (Ubuntu - Free Tier Optimized)
*   **Network:** `DuckDNS` (Dynamic DNS)
*   **Port Management:** Docker Port Mapping (80 -> 8501)
*   **Model Auth:** Hugging Face Token Authentication

## Libraries Used

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Torchaudio](https://img.shields.io/badge/Torchaudio-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![WhisperX](https://img.shields.io/badge/WhisperX-%23000000.svg?style=for-the-badge&logo=openai&logoColor=white)
![Faster-Whisper](https://img.shields.io/badge/Faster--Whisper-%23000000.svg?style=for-the-badge&logo=openai&logoColor=white)
![Pyannote](https://img.shields.io/badge/Pyannote.audio-%23FF9900.svg?style=for-the-badge&logo=huggingface&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Librosa](https://img.shields.io/badge/Librosa-%23dcae9b.svg?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-%23FF9900.svg?style=for-the-badge&logo=amazon-aws&logoColor=white)
![Python-Docx](https://img.shields.io/badge/Python--Docx-%232D3748.svg?style=for-the-badge&logo=microsoftword&logoColor=white)

---

## 📂 Project Structure

*   **`ai/`**
    *   `main.py` → Core logic containing `slow_model` (WhisperX) and `fast_model` (Faster-Whisper + Custom Clustering).
    *   Model loaders and caching mechanisms (`@st.cache_resource`).
*   **`frontend/`**
    *   `the_page.py` → UI layout, file uploader, and mode selection buttons.
*   **`app.py`**
    *   Main entry point for the Streamlit application.
*   **`Dockerfile`**
    *   `python:3.13.3` image with CPU-only PyTorch build.
*   **`requirements.txt`**
    *   List of dependencies optimized for cloud deployment.
*   **`.env`**
    *   Environment variables (Contains Hugging Face Token).

---

## How to Run Locally

You can run the system in **two different ways**:

*   Using **Docker** (recommended for isolation)
*   Running manually with Python

---

## 1. Environment Variable Configuration (.env)

You need a Hugging Face token to access segmentation models (`pyannote/speaker-diarization`). Create a file named `.env` in the project root:

```ini
HUGGING_FACE_TOKEN=hf_your_hugging_face_token_here
```

*Ensure you have accepted the user agreement for `pyannote/speaker-diarization-3.1` on Hugging Face.*

---

## 2. Option A: Run with Docker (Recommended)

This is the **most stable** way to run the project, ensuring all system dependencies (FFmpeg, etc.) are installed.

### Steps

**Build the image:**
```bash
docker build -t speech-app .
```

**Run the container:**
```bash
docker run -p 8501:8501 --env-file .env speech-app
```

Open your browser:
```
http://localhost:8501
```

---

## 3. Option B: Run Without Docker (Manual Setup)

### System Requirements
You must have **FFmpeg** installed on your system.

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Start the Application

```bash
streamlit run app.py
```

Frontend runs at:
```
http://localhost:8501
```

---

## Model Authentication Flow

Since `Pyannote.audio` models are gated, the application requires authentication:

1.  Application starts and looks for `HUGGING_FACE_TOKEN`.
2.  **Fast Mode:** Uses `Faster-Whisper` (No auth required) + Custom KMeans clustering.
3.  **Pro Mode:** Authenticates with Hugging Face via the token to download/load `pyannote/speaker-diarization` models.
4.  If the token is invalid or missing, the Pro model will fail to load.

---

## Production Deployment

*   **Deployed on:** `AWS EC2 (Ubuntu)`
*   **Optimization:** Configured with **Swap Memory (8GB)** to handle model loading on limited RAM.
*   **Docker Optimization:** Uses `torch --index-url .../cpu` to reduce image size by removing CUDA dependencies.
*   **Access:** Served via `DuckDNS` with Port 80 redirection.

---

## Summary

This project is a sophisticated audio analysis tool.
It is a **complete, cloud-deployed AI system** that demonstrates:

*   **Advanced Audio Processing:** Combining ASR and Diarization.
*   **Resource Management:** Running Large Language Models (LLMs) on CPU.
*   **Containerization:** Custom Docker optimization for AI workloads.
*   **Full Stack Integration:** From raw audio processing to user-friendly UI.
