Sentiment Analysis on Social Media Data
This project performs sentiment classification on a social media dataset using text preprocessing, feature engineering, SMOTE for handling class imbalance, and XGBoost for classification.

Dataset
-CSV file: sentimentdataset.csv
-Features: Text, Hashtags, Timestamp, Country, Likes, Retweets, Platform, Sentiment

1. Data Preprocessing
-Removed URLs, mentions, special characters.
-Tokenized, removed stopwords (with exceptions), and applied lemmatization.
-Cleaned hashtags and text separately.
-Generated word cloud and plotted top hashtags.

2. Feature Engineering
-Extracted year, month, day, hour from timestamps.
-Created TF-IDF features for text.
-Encoded categorical variables using one-hot and label encoding.
-CountVectorizer applied on hashtags.
-Combined all features for modeling.

3. Class Imbalance Handling
-Visualized class imbalance.
-Applied SMOTE to oversample minority classes.

4. Model Training
-Model: XGBoost Classifier (multi:softmax)
-Evaluated with:
   -Accuracy
   -Classification Report
   -Confusion Matrix
   
5. Hyperparameter Tuning
-Used RandomizedSearchCV with:
-max_depth, learning_rate, n_estimators, gamma


