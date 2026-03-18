# Data-Driven Personalised Educational Content Recommendation System

## Overview
This project implements a hybrid recommendation system for educational content using the Open University Learning Analytics Dataset (OULAD).

The system analyses learner interaction data and generates personalised recommendations using:

- Content-Based Filtering
- Collaborative Filtering (Matrix Factorisation)
- Hybrid Recommendation Model

## Features
- Data preprocessing pipeline
- Feature engineering using TF-IDF
- Multiple recommendation models
- Cross-validation evaluation
- REST API using FastAPI
- Recommendation demo script

## Dataset
Open University Learning Analytics Dataset (OULAD)

## Data

The project uses the Open University Learning Analytics Dataset (OULAD).

Due to GitHub file size limits, raw and processed datasets are not stored in this repository.
To reproduce the project:

1. Download the OULAD dataset or extract from dataset.zip in repository
2. Place the files in `data/oulad/`
3. Run the preprocessing and training scripts

## Installation

```bash
pip install -r requirements.txt