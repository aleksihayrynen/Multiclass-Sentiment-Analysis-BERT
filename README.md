# Multiclass Sentiment Classification with BERT 🚀

This project implements a multiclass sentiment classification model using BERT. The model was developed and trained entirely on **Google Colab**, leveraging its GPU acceleration for efficient fine-tuning.

---

## **Key Points of the Project** 🔑
- **Deep Learning**: Fine-tuning BERT for sentiment classification.
- **Sentiment Analysis**: Multiclass and grouped-class analysis.
- **Data Visualization**: Evaluating performance metrics and comparing models.

---

## **Initial Goals and Evolution**

The initial goal of this project was to create a **regression model** that could predict a percentage score indicating how positive or negative a sentence or text was. However, as the project evolved, we realized that a **classification-based approach** would be more effective for this task. We then adapted the model's output and usage to fit different needs based on its use case.

---

## **Exploring Sentiment Classes**

Initially, we explored **multiclass sentiment analysis** using five classes to categorize sentiment with greater granularity. However, we found this approach to be quite challenging due to the difficulty in accurately distinguishing between closely related sentiments. To simplify and improve the model's accuracy, we transitioned to using **three broader classes**: **Positive**, **Neutral**, and **Negative**.

---

### **Model Accuracy Comparison**

| Approach           | Number of Classes | Accuracy | Notes                              |
|--------------------|-------------------|----------|------------------------------------|
| **5-class model**  | 5                | 63%      | Difficulty in distinguishing subtle sentiment variations. |
| **3-class model**  | 3                | 81%      | Improved accuracy and generalization. |

---

The project began with simple regression and classification models implemented in **Python**. As the complexity grew, we transitioned to **Google Colab**, leveraging its GPU acceleration for efficient training and fine-tuning.

---




---

## 📑 Table of Contents
1. [🌟 Features](#-features)  
2. [📊 Dataset](#-dataset)  
3. [⚙️ Setup and Installation](#️-setup-and-installation)  
4. [🏋️ Training the Model](#️-training-the-model)  
5. [📈 Evaluation](#️-evaluation)  
6. [🔮 Results](#-results)  
7. [📌 Limitations](#-limitations)  
8. [🔧 Future Work](#-future-work)  
9. [🤝 Contributing](#-contributing)  
10. [📜 License](#-license)  

---

## 🌟 Features
✨ **Fine-tuned BERT** for multiclass sentiment classification.  
✨ **Deep Learning** with Google Colab.  
✨ **Detailed Evaluation Metrics**: Accuracy, Precision, Recall, F1-score, and Confusion Matrix.  
✨ **Example Predictions** for real-world testing.

---

## 📊 Dataset

- **Source**: [Yelp Reviews for SA Fine-Grained 5 Classes on Kaggle](https://www.kaggle.com/datasets/yacharki/yelp-reviews-for-sa-finegrained-5-classes-csv/data)  
- **Total Size**: 650,000 examples  
- **Used Dataset**: 50,000 examples (training, validation, and test sets)  
- **Classes**: Ratings from **1** to **5**  

### **Dataset Details**  

The dataset comes with 600 000 training instance and a seperate test set of 50 000, but for this project, I chose to work with a subset of 50,000 examples from the entire dataset. The data was split into **training**, **validation**, and **test** sets, keeping the splits **stratified** to maintain class balance.

---

### **Class Distributions**  

#### **Train Subset Distribution**
| Sentiment Rating | Count  |
|-------------------|--------|
| 1                | 10,000 |
| 2                | 10,000 |
| 3                | 10,000 |
| 4                | 10,000 |
| 5                | 10,000 |

#### **Validation Subset Distribution**
| Sentiment Rating | Count  |
|-------------------|--------|
| 1                | 2,000  |
| 2                | 2,000  |
| 3                | 2,000  |
| 4                | 2,000  |
| 5                | 2,000  |

#### **Test Subset Distribution**
| Sentiment Rating | Count  |
|-------------------|--------|
| 1                | 2,000  |
| 2                | 2,000  |
| 3                | 2,000  |
| 4                | 2,000  |
| 5                | 2,000  |

---

### **Dataset Shapes**
| Subset      | Number of Samples | Shape    |
|-------------|-------------------|----------|
| **Train**   | 50,000            | (50,000, 2) |
| **Validation** | 10,000         | (10,000, 2) |
| **Test**    | 10,000            | (10,000, 2) |

---

### **Example Data Snippet**  

| Input Text                              | Sentiment Rating |
|-----------------------------------------|------------------|
| "This product exceeded my expectations!"| 5 (Positive)     |
| "It was okay, nothing special."         | 3 (Neutral)      |
| "Terrible service, very disappointed."  | 1 (Negative)     |

---

### **Training Data Split**  
The training data was split while ensuring class stratification. Each subset maintains a balanced distribution of all five sentiment classes. This approach ensures fair performance evaluation and improves model training stability.


---

## ⚙️ Setup and Installation

1. Open the project in Google Colab by clicking the badge below:  
   [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/10cUKwI2EQz7RsqNm2nf_hkW8tqLl_aYF?usp=sharing)

---

### 🗂️ **Loading the Dataset**

To start training, you'll need to load the dataset into the Colab environment. Since the dataset is too large to include directly in the notebook, you will need to connect your **Google Drive** and access the data stored there.

There is already a block in the notebook to connect your Google Drive. Once connected, navigate to the location where your dataset is stored in your Drive and load it into the environment.

1. **Mount Google Drive**
   The block in the notebook will allow you to mount your Google Drive. Once connected, you can access the dataset from your Drive.

2. **Load the Dataset**:
   After mounting Google Drive, use the appropriate path to load the dataset. Example:

   ```python
   # Load train and test datasets
  train_data = pd.read_csv("/content/drive/My Drive/yelp_review_fine-grained_5_classes_csv/train.csv")
  test_data = pd.read_csv("/content/drive/My Drive/yelp_review_fine-grained_5_classes_csv/test.csv")
  ```


### **Training the Model**

```markdown
## 🏋️ Training the Model

To fine-tune the model:

1. Upload your dataset to Colab (follow the steps in the **Setup and Installation** section to load your dataset).
2. Set the training mode:  
   - **True** to train a new model.
   - **False** to load an existing model for inference.

3. Run the training cell:
4. Evaluate your existing model or train a new one

### **Evaluation**

```markdown
## 📈 Evaluation
To evaluate the model, run the evaluation cell:
```python
evaluate(model, tokenizer, test_data)


---

### **Results**

```markdown
## 🔮 Results
### Confusion Matrix

## 📌 Limitations
⚠️ Limited to English text; performance may degrade with non-English text.  
⚠️ Dataset bias may influence predictions if the training data isn't representative of diverse sentiments.  
⚠️ May not generalize well to domains not present in the training data.  

## 🔧 Future Work
🔹 Extend to multilingual sentiment analysis.  
🔹 Optimize for real-time applications.  
🔹 Train on a larger, more diverse dataset for improved generalization.  

## 🤝 Information
This project was made for my OpenSource software course during my exchange in Korean University. This was the first time I got to learn and apply datascience and Artificial intelligence

## 📜 License
This project is licensed under the [MIT License](LICENSE).  



