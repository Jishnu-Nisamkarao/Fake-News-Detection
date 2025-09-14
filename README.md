# Fake News Detection 

This project develops and evaluates several machine learning models to detect fake news articles, utilizing a combined dataset from `gossipcop` and `politifact`. The goal is to build a classification system that can accurately distinguish between real and fake news based on article titles.

## 🌟 The Importance of Fake News Detection

In today's digital age, the rapid spread of misinformation poses a significant threat to society, public trust, and informed decision-making. Fake news can influence public opinion, incite social unrest, and even impact political outcomes. This project addresses this critical issue by leveraging machine learning to automatically identify and flag potentially misleading information. By doing so, it aims to:

* **Combat Misinformation**: Provide a tool to help users and platforms identify and filter out false content.
* **Promote Media Literacy**: Highlight the characteristics that distinguish fake news from real news, aiding in better media consumption habits.
* **Inform a Solution**: Evaluate different machine learning models to determine which is most effective for this classification task.

## 💻 Project Structure & Methodology

This project follows a standard machine learning workflow, from data preprocessing to model evaluation.

### Data Acquisition
The project uses four distinct datasets:
* `gossipcop_fake.csv`
* `gossipcop_real.csv`
* `politifact_fake.csv`
* `politifact_real.csv`

These datasets are combined into a single DataFrame, and a `label` column is added to classify each article as `fake` (0) or `real` (1).

### Data Preprocessing & Cleaning
Before training the models, the data undergoes several cleaning steps to ensure quality:
* **Column Reduction**: Unnecessary columns like `id`, `news_url`, and `tweet_ids` are dropped as they are not relevant for text-based classification.
* **Text Cleaning**: A custom function is used to clean the article titles by:
    * Converting all text to lowercase.
    * Removing URLs, special characters, and punctuation.
    * Eliminating extra whitespace.
* **Missing Values**: Rows with missing titles are handled to prevent errors during vectorization.

### Feature Engineering (Vectorization)
The cleaned text data is converted into a numerical format that machine learning models can understand using **TF-IDF (Term Frequency-Inverse Document Frequency)**. This technique assigns a numerical weight to each word, reflecting its importance in the document and the corpus.

### Model Training & Evaluation
The project evaluates four different classification models to find the best performer:
* **Logistic Regression**
* **Multinomial Naive Bayes**
* **Random Forest Classifier**
* **Linear Support Vector Machine (LinearSVC)**

The dataset is split into training and testing sets, and each model's performance is measured using **Accuracy**, **Precision**, **Recall**, **F1-Score**, and a **Confusion Matrix**.

## 📊 Results & Analysis

The models were evaluated, and their performance metrics were compiled into a summary table. The results reveal interesting insights into their classification abilities, particularly in correctly identifying fake news.

| Model | Accuracy | Precision (Fake) | Recall (Fake) | F1 (Fake) | Precision (Real) | Recall (Real) | F1 (Real) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LinearSVM** | **0.8297** | **0.7319** | **0.4922** | **0.5885** | 0.8493 | 0.9407 | 0.8927 |
| **Logistic Regression** | 0.8289 | 0.7565 | 0.4547 | 0.5680 | 0.8415 | 0.9519 | **0.8933** |
| **Naive Bayes** | 0.8203 | 0.7574 | 0.4024 | 0.5256 | 0.8298 | 0.9576 | 0.8891 |
| **Random Forest** | 0.7888 | 0.8652 | 0.1733 | 0.2888 | 0.7848 | 0.9911 | 0.8760 |

### Key Observations:
* **Overall Accuracy**: LinearSVM and Logistic Regression achieved the highest overall accuracy, demonstrating strong performance across the dataset.
* **Random Forest**: This model showed very high precision for fake news but extremely low recall. This means it is very cautious and only labels an article as "fake" when it is highly certain, missing many false positives.
* **LinearSVM**: This model offered the best balance between precision and recall for detecting fake news. Its ability to correctly identify a larger proportion of fake articles (higher recall) makes it a reliable choice for this task.
* **Real News Detection**: All models performed exceptionally well in identifying real news articles, with high recall scores across the board.

## Conclusion

The analysis shows that **LinearSVC** and **Logistic Regression** are the most effective models for this fake news detection task. While Logistic Regression achieved a slightly better F1 score for real news, LinearSVC's superior recall on fake news makes it a more compelling choice for a system where correctly identifying misinformation is the primary objective.
