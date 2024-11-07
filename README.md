# Text Processing and Tokenization with BKit

This project provides a Python script that processes text data from a CSV file. It applies various text transformations including tokenization, lemmatization, named entity recognition (NER), part-of-speech (POS) tagging, and digit cleaning. The processed data is saved to a new CSV file.

## Requirements


```bash
pip install pandas bkit[all] 
```
## **Usage**

```bash
python main.py input_csv output_csv

```
#### Arguments:
`input_csv:` Path to the input CSV file containing the text column (text).
`output_csv: `Path to the output CSV file where the processed data will be saved.
