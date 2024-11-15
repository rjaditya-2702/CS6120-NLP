from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
# from datasets import Dataset, DatasetDict

dataset_path = "text.csv"
df = pd.read_csv(dataset_path)
df = df.drop("Unnamed: 0", axis=1)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=10, shuffle=True)

from transformers import GPT2Tokenizer
from transformers import GPT2ForSequenceClassification, GPT2Config

configuration = GPT2Config()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2ForSequenceClassification(configuration).from_pretrained("gpt2", num_labels=6)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

from datasets import Dataset, DatasetDict

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

train_tokenized_datasets = train_dataset.map(tokenize_function, batched = True)
test_tokenized_datasets = test_dataset.map(tokenize_function, batched = True)

train_tokenized_datasets = train_tokenized_datasets.remove_columns(['text',  '__index_level_0__'])
test_tokenized_datasets = test_tokenized_datasets.remove_columns(['text',  '__index_level_0__'])

train_tokenized_datasets.set_format(type='torch', columns = ['input_ids', 'attention_mask', 'label'])
test_tokenized_datasets.set_format(type='torch', columns = ['input_ids', 'attention_mask', 'label'])

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized_datasets,
    eval_dataset=test_tokenized_datasets,
)

trainer.train()

model.save_pretrained("model")
tokenizer.save_pretrained("model")

trainer.save_model('model2')
trainer.model.save_pretrained('model3')