from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd


def train_logistic_model(df: pd.DataFrame):
    """
    Train and evaluate a Logistic Regression model with balanced class weights.
    
    Parameters:
        df (pd.DataFrame): Preprocessed dataframe with 'clean_text' and 'label'.
    
    Returns:
        None (prints evaluation metrics)
    """

    # Create TF-IDF features from clean text
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["clean_text"])  # Features
    y = df["label"]                                 # Labels

    # Split dataset into training (80%) and test (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Instantiate Logistic Regression with class weights balanced
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_train, y_train)  # Train the model

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    print("Model Evaluation:")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    return model, vectorizer

def predict_sentiment(model, vectorizer, texts: list[str]) -> list[str]:
    """
    Predict the sentiment for a list of texts.

    Parameters:
        model: Trained sklearn model.
        vectorizer: TF-IDF vectorizer.
        texts (list[str]): List of raw news headlines.

    Returns:
        list[str]: Predicted sentiment labels.
    """
    from src.preprocessing import clean_text

    cleaned = [clean_text(t) for t in texts]             # Preprocess
    X = vectorizer.transform(cleaned)                    # Vectorize
    preds = model.predict(X)                             # Predict
    return preds.tolist()
