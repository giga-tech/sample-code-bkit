import argparse
import bkit
import pandas as pd

# Initialize models
ner = bkit.ner.Infer('ner-noisy-label')
pos = bkit.pos.Infer('pos-noisy-label')

def tokenize_csv(input_csv, output_csv):
    """
    Tokenizes and processes text data from the input CSV file and saves the results to an output CSV file.
    Args:
        input_csv (str): Path to the input CSV file containing a 'text' column.
        output_csv (str): Path to the output CSV file where the processed data will be saved.
    """

    df = pd.read_csv(input_csv)
    
    # Check if 'text' column exists
    if 'text' not in df.columns:
        raise ValueError("Input CSV must contain a 'text' column.")
    
    # Apply the tokenizer to the 'text' column
    df['tokenized'] = df['text'].apply(lambda x: bkit.tokenizer.tokenize(x))
    df['Clear text'] = df['text'].apply(lambda x: bkit.transform.clean_text(x))
    df['Lemmatization'] = df['text'].apply(lambda x: bkit.lemmatizer.lemmatize(x))
    df['Clear digit'] = df['text'].apply(lambda x: bkit.transform.clean_digits(x))
    df['NER'] = df['text'].apply(lambda x: ner(x))
    df['POS'] = df['text'].apply(lambda x: pos(x))
    df["word_punctuation_tokenizer"] = df['text'].apply(lambda x: bkit.tokenizer.tokenize_word_punctuation(x))

    # Save the resulting DataFrame to a new CSV file
    df.to_csv(output_csv, index=False)
    print(f"Output data saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a CSV file with text and apply tokenization, NER, POS, and other transformations.")
    parser.add_argument('input_csv', type=str, help="Path to the input CSV file.")
    parser.add_argument('output_csv', type=str, help="Path to the output CSV file.")
    
    args = parser.parse_args()
    
    tokenize_csv(args.input_csv, args.output_csv)
