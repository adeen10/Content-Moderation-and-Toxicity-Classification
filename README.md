# Content Moderation and Toxicity Classification

## Overview

This project focuses on content moderation and toxicity classification using the Jigsaw Toxic Comment Classification Dataset. The goal is to identify various forms of comment toxicity, including mild toxicity, severe toxicity, obscenity, threats, insults, and identity hate. Three different approaches were employed:

1. Naive Bayes and Logistic Regression
2. Sequence Model - Recurrent Neural Network (RNN)
3. Pretrained Encoder-Transformer - BERT

## Clean Vs. Toxic Comments Imbalance in Training Dataset

To address the class imbalance between "clean" and "toxic" comments in the original dataset, a strategic approach was implemented. A balanced dataset was created by sampling a subset of 16,225 clean comments and merging them with existing toxic comments.

### Class Imbalances in Training Dataset

The "toxic" class had the highest number of samples, while "severe toxic," "threat," and "identity hate" had significantly fewer samples. Different techniques were employed in each model to mitigate these imbalances.

## Naive Bayes and Logistic Regression

### Methodology

1. **Data Preprocessing:** Lowercasing, punctuation removal, lemmatization, and handling of non-ASCII characters were performed on the dataset.
2. **Feature Extraction:** TF-IDF was chosen for its efficiency and interpretability.
3. **Class Imbalance Handling:** Random sampling and class weights were used to address the imbalance issue.

### Results

Naive Bayes showed an F1-score of 0.857 for 'toxic' comments and as low as 0.034 for 'severe_toxic'. Logistic Regression exhibited higher F1-scores across all categories, indicating better balance between precision and recall.

### Analysis

Logistic Regression generally achieved higher Recall, F1-score, and Accuracy across multiple toxicity categories compared to Naive Bayes, especially in scenarios requiring interpretability of model predictions.

## Sequence Model - Recurrent Neural Network (RNN)

### Methodology

1. **Data Preprocessing:** Tokenization and sequence generation were performed using Keras's Tokenizer class.
2. **RNN Model Architecture:** An Embedding Layer, SimpleRNN Layer, and Dense Layer were used to capture sequential patterns and classify comments.
3. **Training and Evaluation:** Class weights were applied during training to handle class imbalance.

### Results and Analysis

The RNN achieved an overall test accuracy of 89.61%. However, it struggled with classes having fewer instances. Further refinements are needed to enhance its performance, especially in handling class imbalance and capturing context for minority classes.

## Pre-Trained Encoder Transformer - BERT

### Methodology

1. **Data Preprocessing:** Label summarization, data separation and sampling, and shuffling were performed.
2. **Tokenization and Encoding:** The BERT model was fine-tuned using tokenized and encoded input with attention masks.
3. **Model Training:** The AdamW optimizer was employed with a learning rate of 2e-5.

### Results and Analysis

The BERT model demonstrated an accuracy of 74.83%, with a high recall rate of 88.29%. While precision was lower, the model maintained a balanced trade-off between precision and recall, as evidenced by its robust F1 score of 54.93%.

## Model Comparison

Considering the results and analysis, BERT Transformer emerges as the most promising model. Its high recall rate is valuable in toxic comment classification, and its advanced language understanding capabilities give it an edge.

## Future Directions

- **Model Ensembling:** Combining the strengths of BERT with the precision of Logistic Regression in a hybrid model could offer more balanced performance.
- **Extended Dataset and Augmentation:** Including a more diverse and extensive dataset, possibly augmented with synthetic data, can improve model training, especially for underrepresented categories.
- **Managing Class Imbalance:** Employing more sophisticated techniques like SMOTE or adaptive resampling methods can provide better results in handling class imbalance.
