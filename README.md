# 🎓 Data-Driven Personalised Educational Content Recommendation System

## 📌 Overview

This project implements a **data-driven recommendation engine** for personalised educational content using the **Open University Learning Analytics Dataset (OULAD)**.

The system processes student interaction data and applies multiple machine learning techniques to generate personalised recommendations.

The project was developed as part of a final-year Computer Science dissertation under:

> **Template 1.1 – Data-Driven Personalised Educational Content Recommendation**

---

## 🚀 Key Features

* Data preprocessing pipeline for large-scale educational datasets
* Multiple recommendation models:

  * Popularity-based baseline
  * Content-based filtering
  * Matrix Factorisation (Collaborative Filtering)
  * Hybrid recommender system
* Cross-validation evaluation framework
* REST API for real-time recommendations
* Command-line demo for user-level recommendations
* Exportable recommendation results (CSV + JSON)

---

## 🧠 Technologies Used

* Python 3.12
* Pandas / NumPy
* Scikit-learn
* Joblib
* FastAPI (API layer)
* Uvicorn (server)

---

## 📂 Project Structure

```
final-recommender/
│
├── api/                    # FastAPI application
├── src/                    # Core source code
│   ├── preprocess.py
│   ├── train_models.py
│   ├── experiments.py
│   ├── demo_recommend.py
│   └── download_data.py
│
├── data/
│   ├── oulad/              # Raw dataset (not included)
│   ├── raw/
│   └── processed/
│
├── models/                 # Saved trained models
├── recommendations/        # Output results
├── notebooks/              # EDA + training notebooks
└── README.md
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository

```bash
git clone https://github.com/AidanBhags/educational-recommender-system.git
cd educational-recommender-system
```

---

### 2️⃣ Create and activate virtual environment (Recommended to avoid dependency conflicts. The project can also run using a global Python environment.)

#### Windows (PowerShell)

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset Setup

This project uses the **Open University Learning Analytics Dataset (OULAD)**.

### ✅ Option 1 (Recommended – Automatic)

```bash
python src/download_data.py
```

---

### ⚠️ Option 2 (Manual Download)

Download from:
👉 https://drive.google.com/file/d/1CJFqqyYQ8eDTUJJrd7qPx5-8Jl66lVcw/view

Then place the files inside:

```
data/oulad/
```

---

## 🔄 Running the Pipeline

### 1️⃣ Preprocess data

```bash
python src/preprocess.py
```

---

### 2️⃣ Train models

```bash
python src/train_models.py
```

---

### 3️⃣ Run evaluation (cross-validation)

```bash
python src/experiments.py
```

---

### 4️⃣ Run recommendation demo

```bash
python src/demo_recommend.py
```

Example:

```
Enter user_id: 28400
```

Outputs:

* User history
* Content-based recommendations
* Hybrid recommendations
* CSV + JSON saved in `/recommendations/`

---

## 🌐 Running the API

Start the API server:

```bash
uvicorn api.app:app --reload
```

Open in browser:

```
http://127.0.0.1:8000/docs
```

---

### Example API Request

```
GET /users/{user_id}/history
GET /recommend/content/{user_id}
GET /recommend/hybrid/{user_id}
```

---

## 📈 Evaluation Metrics

The system is evaluated using:

* Precision@10
* Recall@10
* NDCG@10
* RMSE (for collaborative filtering)
* Cross-validation (3-fold)

---

## 🧪 Example Results (Cross-Validation)

| Model      | Precision@10 | Recall@10   | NDCG@10     |
| ---------- | ------------ | ----------- | ----------- |
| Popularity | ~0.0176      | ~0.0114     | ~0.0187     |
| Content    | ~0.0963      | ~0.0761     | ~0.1074     |
| MF         | ~0.0171      | ~0.0086     | ~0.0213     |
| Hybrid     | **~0.0961**  | **~0.0759** | **~0.1073** |

---

## 📊 Outputs

The system produces:

* Trained models (`models/`)
* Processed datasets (`data/processed/`)
* Recommendation outputs:

  * CSV files
  * JSON summaries
* API responses with recommendation scores

---

## 📓 Notebooks

* `01_eda.ipynb` → Exploratory Data Analysis
* `02_train_eval.ipynb` → Model training & evaluation

---

## ⚠️ Notes

* Large datasets are not included due to GitHub limits
* Use the provided script or link to download data
* `.gitkeep` files preserve folder structure

---

## 🔗 Repository

GitHub:
👉 https://github.com/AidanBhags/educational-recommender-system

---

## 🎥 Demo Video

A 3–5 minute demonstration video is included in the submission, showcasing:

* Data pipeline
* Model training
* Recommendation outputs
* API functionality

---

## 📚 References

(Full references provided in report submission)

---

## 🏁 Final Remarks

This project demonstrates:

✔ End-to-end data science pipeline
✔ Multiple recommendation techniques
✔ Hybrid model outperforming baselines
✔ Real-world deployment via API

It satisfies the requirements of a **data-driven recommendation system** with **robust evaluation and reproducibility**.

---
