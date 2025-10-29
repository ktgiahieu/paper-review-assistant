import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
import numpy as np
from seqeval.metrics import classification_report
import ast # Added for safely evaluating string representations of lists

# --- 1. Define NER labels and mappings ---
# We use the standard IOB (Inside, Outside, Beginning) format.
label_list = ["O", "B-FLAW", "I-FLAW"]
label_to_id = {label: i for i, label in enumerate(label_list)}
id_to_label = {i: label for i, label in enumerate(label_list)}
NUM_LABELS = len(label_list)


# --- 2. Define a Custom NER Dataset ---
class FlawNERDataset(Dataset):
    """PyTorch Dataset for the flaw NER task."""
    def __init__(self, df, tokenizer, max_length=512):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row['review_text']
        spans = row['spans'] # This is now a list of (start_char, end_char) tuples

        # Tokenize the text. Padding is now handled by the data collator.
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True # Needed to map char spans to tokens
        )

        offset_mapping = encoding.pop("offset_mapping")
        num_tokens = len(encoding['input_ids'])

        # Initialize labels for the actual number of tokens.
        labels = [label_to_id["O"]] * num_tokens

        # Iterate over the flaw spans for this review
        for start_char, end_char in spans:
            # Find the start and end tokens for the character span
            start_token_idx = -1
            end_token_idx = -1

            for i, (offset_start, offset_end) in enumerate(offset_mapping):
                # We need to handle tokens that don't map to characters (e.g., [CLS], [SEP])
                if offset_end == 0:
                    continue

                # Find the start token index
                if start_token_idx == -1 and offset_start <= start_char < offset_end:
                    start_token_idx = i

                # Find the end token index
                # The end token is the one that contains the end_char
                if end_token_idx == -1 and offset_start < end_char <= offset_end:
                    end_token_idx = i
            
            # If the span was found within the tokenized sequence
            if start_token_idx != -1 and end_token_idx != -1:
                # Label the first token as B-FLAW
                labels[start_token_idx] = label_to_id["B-FLAW"]
                # Label all subsequent tokens in the span as I-FLAW
                for i in range(start_token_idx + 1, end_token_idx + 1):
                    # Ensure we don't go out of bounds for labels
                    if i < len(labels):
                        labels[i] = label_to_id["I-FLAW"]

        # The Trainer expects 'labels' as a key
        encoding["labels"] = labels
        return encoding


# --- 3. Compute Metrics for NER Evaluation ---
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Convert IDs to labels, removing special tokens (-100 which Hugging Face uses for padding)
    true_predictions = [
        [id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id_to_label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Use seqeval to get a detailed report
    report = classification_report(true_labels, true_predictions, output_dict=True)

    # Return key metrics, checking if 'FLAW' key exists
    if "FLAW" in report:
        return {
            "precision": report["FLAW"]["precision"],
            "recall": report["FLAW"]["recall"],
            "f1": report["FLAW"]["f1-score"],
        }
    else: # Handle cases with no predicted FLAW entities
        return { "precision": 0.0, "recall": 0.0, "f1": 0.0 }


# --- 4. Main Training Script ---
if __name__ == '__main__':
    # Load the processed data
    try:
        df = pd.read_csv('labeled_flaws_ner.csv')
    except FileNotFoundError:
        print("Error: 'labeled_flaws_ner.csv' not found. Please run 'label_data.py' first.")
        exit()

    # The 'spans' column is stored as a string. Safely convert it back to a list of tuples.
    df['spans'] = df['spans'].apply(ast.literal_eval)

    # The data is already in the correct format (one row per review with a list of spans).
    # We can split it directly.
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)

    # Initialize tokenizer and a pre-trained token classification model
    MODEL_NAME = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=id_to_label,
        label2id=label_to_id
    )

    # Create datasets
    train_dataset = FlawNERDataset(train_df, tokenizer)
    eval_dataset = FlawNERDataset(eval_df, tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results_ner',
        num_train_epochs=10,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs_ner',
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
    )

    # Initialize the Data Collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Start training
    print("Starting NER model training...")
    trainer.train()

    # Evaluate the model
    print("\nEvaluating the final NER model...")
    eval_results = trainer.evaluate()
    print("Evaluation results:", eval_results)


