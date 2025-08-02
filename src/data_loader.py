import pandas as pd  # Pandas is the main library for handling dataframes

def load_raw_news(path: str) -> pd.DataFrame:
    """
    Loads the raw financial news dataset from a CSV file.
    
    Parameters:
        path (str): The file path to the CSV file.
    
    Returns:
        pd.DataFrame: A DataFrame with 'label' and 'text' columns.
    """

    # Read the CSV file with the correct encoding (Windows files often use ISO-8859-1)
    df = pd.read_csv(
        path,
        encoding='ISO-8859-1',       # Change encoding to avoid decode errors
        names=["label", "text"],     # Define the column names explicitly
        header=None                  # We don't use the first row as a header
    )

    # Drop rows that are completely empty
    df = df.dropna()

    # Clean up whitespace and lowercase the label column
    df["text"] = df["text"].str.strip()              # Remove extra spaces
    df["label"] = df["label"].str.lower().str.strip()  # Standardize labels

    return df
