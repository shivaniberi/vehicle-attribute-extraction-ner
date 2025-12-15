# Named Entity Recognition for Vehicle Attribute Extraction
Named Entity Recognition (NER) system for extracting fine-grained vehicle attributes from natural language descriptions using RNN, LSTM, BiLSTM, BERT, and RoBERTa models. Includes full data preprocessing, model training, evaluation, and performance comparison on the FindVehicle dataset. This repository contains a complete Named Entity Recognition (NER) pipeline for extracting fine-grained vehicle attributes from natural-language descriptions. The project evaluates multiple deep learning architectures, ranging from classical recurrent neural networks to transformer-based models, using the FindVehicle dataset. RoBERTa achieved the best performance across all models tested.

---

## Overview

Accurate extraction of vehicle attributes such as brand, model, color, type, orientation, and location is essential for natural-language-based vehicle retrieval systems. The FindVehicle dataset provides detailed annotations that enable the training of high-quality NER models.
This project includes data preprocessing, model training, evaluation, and performance comparison between RNN, LSTM, BiLSTM, BERT, and RoBERTa architectures.
A detailed report of the project is included in this repository. 

---
![Architecture](https://github.com/user-attachments/assets/653d5618-c14e-4db5-83ad-a1295c342936)

## Models Implemented

### Recurrent Neural Network Models

* Simple RNN
* LSTM
* BiLSTM

### Transformer-Based Models

* BERT-base
* RoBERTa-base (best performing model)

---

## Performance Summary

| Model     | Precision | Recall | F1-Score | Accuracy |
| --------- | --------- | ------ | -------- | -------- |
| RNN       | 0.4356    | 0.6438 | 0.5196   | 0.6438   |
| LSTM      | 0.4551    | 0.3997 | 0.3967   | 0.7390   |
| BiLSTM    | 0.8510    | 0.8490 | 0.8528   | 0.8810   |
| BERT-base | 0.5567    | 0.6950 | 0.6182   | 0.6587   |
| RoBERTa   | 0.9530    | 0.9603 | 0.9566   | 0.9813   |

RoBERTa demonstrated the highest overall performance.

---
<img width="1451" height="538" alt="image" src="https://github.com/user-attachments/assets/79796998-9b84-49c4-90cf-eccbd38e286b" />


## Project Structure

```
├── data/                                 # Dataset files
├── data_preprocessing/                   # Dataset preprocessing scripts and files
├── Roberta_implementation/               # RoBERTa training notebooks and saved model
├── RNN_implementation.ipynb              # RNN implementation
├── LSTM_BiLSTM_implementation.ipynb      # LSTM and BiLSTM implementations
├── BERT_Implementation_vehicle_dataset.ipynb  # BERT implementations
├── README.md                             # Project documentation
```

---

## Methodology

### Data Preprocessing

* Conversion of CoNLL-style annotations into structured formats
* BIO/BIE tag normalization and cleaning
* Analysis of tag distribution, sentence length, and entity density
* Conversion to HuggingFace `DatasetDict` for transformer compatibility
* Subword alignment strategies for transformer models

### Model Training

* Implementation of multiple architectures under identical training conditions
* Adam and AdamW optimizers with tuned learning rates
* Early stopping and best-model checkpointing
* Loss masking for subword tokens in transformer models

### Evaluation

* Token-level and entity-level evaluation using `seqeval`
* Span-based matching with Jaccard similarity for test set evaluation
* Comparison across recurrent and transformer architectures

---

## Key Findings

* RoBERTa achieved an F1-score of approximately 0.9566 and an accuracy of 0.9813, making it the most effective model for vehicle attribute extraction.
* Recurrent architectures such as RNN and LSTM struggled with multi-word and overlapping entity structures.
* BiLSTM performed significantly better than RNN and LSTM but still lagged behind transformer-based approaches.
* Transformer models, particularly RoBERTa, captured long-range context and domain-specific vocabulary far more effectively.

---

## Dataset

The FindVehicle dataset includes over 42,000 vehicle descriptions and more than 200,000 labeled entities, covering attributes such as:

* Color
* Brand
* Model
* Vehicle type
* Orientation
* Location

It is provided in both CoNLL and JSONLines formats and is designed to support fine-grained NER and relation extraction tasks.

---

## Running the Code

Install dependencies:

```
pip install torch transformers datasets seqeval pandas numpy scikit-learn
```

To train the RoBERTa model, open and run the notebook located in:

```
Roberta code and model/
```

All other models can be trained by running the corresponding notebooks in the root directory.

---

## Future Work

* Joint modeling for NER and relation extraction
* Experiments with larger transformer architectures such as RoBERTa-large or DeBERTa
* Deployment of the NER system for real-time vehicle retrieval applications
* Integration with multimodal retrieval systems (text-to-image)

---

