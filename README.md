# Predicting Amazon Book Ratings using Spark NLP and BERT

This repository contains a scalable sentiment analysis pipeline designed to process large-scale Amazon Book Reviews. Built with **Spark NLP** and **BERT**, the project is optimized for both High-Performance Computing (HPC) environments and local development using **Docker**.

---

## Features
- **Scalable Big Data Processing:** Uses Apache Spark 3.5.0 to handle massive datasets.
- **Deep Learning with BERT:** Leverages `sent_small_bert_L8_512` for state-of-the-art text embeddings.
- **Smart Storage Management:** Automatically detects and extracts `.tsv.zip` datasets if the raw file is missing.
- **Lightweight Dockerization:** Utilizes **Volume Mounting** to keep Docker images small by keeping heavy data and JAR files on the host machine.
- **Robust Preprocessing:** Includes HTML cleaning, noise reduction, and "spoiler" word filtering.

---

## Tech Stack
- **Spark:** 3.5.0
- **Spark NLP:** 5.5.1
- **Java:** 17 (OpenJDK)
- **Python:** 3.10
- **Model:** BERT (Small BERT L8_512)
- **Containerization:** Docker / Apptainer

---

## Project Structure
```text
.
├── big_data.py                  # Main PySpark & NLP script
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Minimal runtime environment
├── spark-nlp-assembly-5.5.1.jar # Spark NLP Fat JAR (Mount via Volume)
└── amazon_reviews_us_Books_v1_02.tsv.zip # Dataset (Mount via Volume)
```

## 🚀 Getting Started

### 1. Prerequisites
- [Docker](https://www.docker.com/) installed on your machine.
- The Amazon dataset (`amazon_reviews_us_Books_v1_02.tsv.zip`) placed in the root directory.
- The Spark NLP assembly JAR (`spark-nlp-assembly-5.5.1.jar`) placed in the root directory.

### 2. Build the Docker Image
The image is designed to be a "slim" runtime environment, containing only the OS, Java 17, and Python dependencies.
```bash
docker build -t amazon-bert-nlp .
```

### 3. Run with Volume Mounting
To keep the setup efficient, we mount the current directory to the container's /app folder. This allows the script to access the dataset and JAR file from your host machine and save the trained model back to your local storage.
```bash
docker run -it --name amazon_nlp_run \
  -v "$(pwd)":/app \
  -e SPARK_MEM=16g \
  amazon-bert-nlp
```
Note: Use -e SPARK_MEM to adjust the RAM allocation based on your hardware (e.g., 32g, 64g).

## How It Works

The system follows a robust data engineering and machine learning workflow to handle large-scale text data:



1.  **Smart Extraction & Verification:** The script first checks if the raw `.tsv` file exists in the working directory. If missing, it automatically extracts it from the `.zip` archive. This minimizes manual setup and saves disk space during transport.

2.  **Spark NLP Pipeline Architecture:**
    The core of the analysis is a modular pipeline:
    - **DocumentAssembler:** Entry point that converts raw text into Spark NLP's `Document` format.
    - **SentenceDetector:** Segments text into individual sentences for more granular BERT analysis.
    - **BertSentenceEmbeddings:** Utilizes the `sent_small_bert_L8_512` model to generate high-quality 512-dimensional vector representations of sentences.
    - **ClassifierDL:** A TensorFlow-based Deep Learning classifier trained on the BERT embeddings to predict star ratings (1 to 5).



3.  **Data Persistence & Optimization:** To prevent Spark's "lazy evaluation" from recalculating the heavy BERT embeddings multiple times, we use `persist(StorageLevel.MEMORY_AND_DISK)`. An explicit `count()` call forces the data into RAM/Disk before the training starts, ensuring stability and performance.

---

## Evaluation & Results

The script splits the data into **80% training** and **20% testing**. Upon completion, it generates a comprehensive **Classification Report** including:

- **Precision:** Accuracy of positive predictions.
- **Recall:** Ability to find all positive instances.
- **F1-Score:** The harmonic mean of precision and recall.
- **Overall Accuracy:** Performance across all star rating classes.



---

