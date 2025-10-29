# EHR Contrastive Learning and Subtyping

This repository contains code to apply contrastive learning on Electronic Health Record (EHR) data and generate patient subtypes. The approach leverages a Transformer-based model fine-tuned using a contrastive loss on sequential EHR data. The code computes patient embeddings, clusters them using KMeans, and calculates the optimal k using prediction strength.

> **Note:** Due to data privacy restrictions, the original vocabulary and model files used in our research cannot be shared. In this repository, a Hugging Face BERT model is used as a placeholder. Please replace it with your own model and vocabulary if necessary.

## Overview

- **finetune.py**  
  Fine-tunes a Transformer-based model on EHR data using contrastive learning.

- **evaluate.py**  
  Generates patient subtypes by computing embeddings with the trained model, clustering them via KMeans, and producing visualizations (e.g., t-SNE plots).

## Requirements

- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- scikit-learn
- pandas
- matplotlib
- seaborn
- MulticoreTSNE

Install the dependencies with:

```bash
pip install torch transformers scikit-learn pandas matplotlib seaborn MulticoreTSNE

## Usage

### Fine-Tuning the Model

Run the following command to fine-tune the model on your EHR data:

```bash
python finetune.py --disease AD --cohort_dir AD_data --experiment_dir AD --model_name cl_maskage_b16 --batch_size 16 --seed 12345 --device 1
```

#### Parameter Explanations:

- `--disease`: Specifies the disease label for training.
- `--cohort_dir `: Directory containing the  cohort data.
- `--experiment_dir `: Directory to save the experiment outputs, including checkpoints and logs.
- `--model_name `: The designated name for the trained model.
- `--batch_size 16`: Batch size used during training.
- `--seed 12345`: Seed value for reproducibility.
- `--device 1`: GPU device identifier for training.

### Evaluating the Trained Model and Generating Subtypes
For around 40k patients, it cost 4 hours on a single V100 node.

After training, generate patient subtypes using the evaluation script:

```bash
python evaluate.py --disease AD --cohort_dir AD_data --model_name AD_model --experiment_dir AD --k 5 --fold_idx 4 --device 0 
```

This code is adapted from https://github.com/Zhengxian-Fan/HF-Subtyping


