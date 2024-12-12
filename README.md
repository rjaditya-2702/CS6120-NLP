## Usage

1. Clone the repository
2. Install the following dependencies:
   - NumPy
   - torch
   - Sklearn
   - transformers
   - evaluate
   - nnsight   
   - PySpark
   - Pandas
   - matplotlib
   - seaborn

### To get the model:
The models are available here - https://drive.google.com/drive/folders/1D7eiUCLUJFTEy0FTlyFbQirYR2dbRhTw?usp=drive_link .

There are two folders - 
1. `model/` - used for all predictions, run tests, etc.
2. `model_with_tokenizer/` - used primarily in token_analysis

Ensure the two models are inside `code/`

### To run the head masking experiment:

1. Run head_masking_samples.py to obtain the samples (the results from this are stored in data/samples if you do not want to install PySpark and run the file.)
2. Run head_masking.py to process the sentences obtained from step1 on each variant. The results are stored as npy files in the execution directory under the names `head_masking_class_prob_diff.npy` and `multi_head_masking_class_prob_diff.py`

### To run the word flipping experiment:

1.  Connect to a python/ ipynb kernel that has the libraries mentioned in the requirements.
2. Run the first five cells.
3. The remaininder of the notebook compares the heatmaps of two sentences. Feel free to experiment by giving custom sentences.

### To run the word replacement experiment:

1. Run `common_words.py` to obtain the 10 most common words per class and replace them with synonyms (the results from this are stored in data/replaced_words_text.csv if you do not want to install PySpark and run the file.)
2.

## Nonstandard dependencies:

- PySpark for sample collection
