# Multimodal Visual Question Answering using the ABO Dataset
## Project Directory Structure
```text
├── Evaluation/
│ ├── evaluation_BaselineModels.ipynb # Evaluation of baseline models
│ └── evaluation_FinetunedModel.ipynb # Evaluation of fine-tuned models
│
├── Inference/
│ ├── inference_baseline_QWEN2B.ipynb # Inference using QWEN 2B baseline
│ └── inference_baseline_QWEN7B.ipynb # Inference using QWEN 7B baseline
│
├── Results/
│ ├── results_qwen2B.csv # Output results from QWEN 2B
│ ├── results_QWEN7B.csv # Output results from QWEN 7B
│ ├── results_QWEN_finetuned_rank4.csv # Fine-tuned (rank 4) results
│ └── results_QWEN_finetuned_rank16.csv # Fine-tuned (rank 16) results
│
├── SFT/ # Finetuning Folder
│ ├── model/qwen_qlora_4/ # Final fine-tuned model (LoRA rank 4)
│ ├── qwen_2.5_low_rank_checkpoints/ # Intermediate checkpoints from training
│ ├── qwen_training_r_4.ipynb # LoRA rank 4 training notebook
│ └── qwen_training_r_16.ipynb # LoRA rank 16 training notebook
│
├── dataset/ # VQA dataset and processing notebooks
│ ├── VQA-dataset/ # Raw VQA CSVs (unsplit)
│ ├── VQA-dataset-train/ # Training split CSVs
│ ├── VQA-dataset-test/ # Test split CSVs
│ ├── data_exploration.ipynb # Dataset analysis and insights
│ ├── data_splitter.ipynb # Script for splitting dataset
│ └── dataset_curation_*.ipynb # Curation notebooks for each listing
│
├── inference_script/
│ ├── inference.py # Python script to run inference via CLI
│ └── requirements.txt # Python environment dependencies
│
├── README.md # Main README file (this file)
└── README_for_inference.md # Separate README for inference usage
```

##  Methodology

###  Data Curation

#### Overview
The task of data curation involved generating a high-quality Visual Question Answering (VQA) dataset from the **Amazon-Berkeley Objects (ABO)** dataset. Our goal was to create diverse, single-word answerable questions based solely on image content using generative models, and prepare a clean train-test split for downstream multimodal learning.

#### Tools and Frameworks Used

- **Gemini 2.0 API** (`google.generativeai`)  
  - Used for multimodal prompt-based question-answer generation.
  - Enabled contextual understanding of images via descriptive and interrogative prompt engineering.
  - API key configured via environment variables.

- **Pandas & NumPy**  
  - Used for metadata handling and dataset processing.

- **PIL (Python Imaging Library)**  
  - Used to validate image integrity and perform resizing.

- **Scikit-learn** (`train_test_split`)  
  - Used for creating training and testing splits from curated CSVs.

- **Python File I/O**  
  - Automated listing-level processing from:
    - `abo-images-small/images/small`
    - `listings/metadata/`

---

### Data Curation Workflow

#### Metadata Extraction

- Parsed individual `listings_x.json` files.
- Mapped product metadata to image metadata via `images.csv` using image IDs.

#### Prompt Engineering

- Structured prompts created using metadata fields like `category`, `style`, and `color`.
- Questions focused on:
  - **Object identification** (e.g., *What is the object?*)
  - **Attribute recognition** (e.g., *What is the color?*)
  - **Style/context inference** (e.g., *Is this modern or vintage?*)
- Prompts passed to **Gemini 2.0 API** to generate questions and single-word answers.

#### CSV Creation and Cleaning

- Each listing generated a separate CSV file with:
  - `image_path`
  - `question`
  - `answer`
- Null records and duplicates were removed for consistency.

---

### Data Splitting Process

- **Per-listing train/test splits** were created using a 90:10 ratio via `train_test_split`.
- **Merged training dataset** was formed by combining listing-wise CSVs.
- The merged data was:
  - **Shuffled**
  - **Re-indexed**
  - **Cleaned of unused columns**

#### Output Files

- Final dataset files:
  - `dataset/VQA-dataset-train/merged_listings_train.csv`
  - `dataset/VQA-dataset-test/merged_listings_test.csv`

---

## Baseline Evaluation

### Objective

To benchmark performance, we evaluated off-the-shelf VQA models on our curated dataset **without any fine-tuning**.

### Models Used

| Model         | Description                                                   |
|---------------|---------------------------------------------------------------|
| **Qwen-VL 7B**| Multimodal transformer with high vision-language capability.  |
| **Qwen-VL 2B**| Smaller variant optimized for low-latency inference.          |


All models were evaluated in **zero-shot** or **few-shot** settings.

---

###  Evaluation Setup

- **Input**: Image + Question → Predicted Answer  
- **Metrics Used**:
  - **Exact String Match**: Case-insensitive exact match.
  - **BERTScore (F1)**: Semantic similarity via contextual embeddings.
  - **WUP Similarity**: WordNet-based semantic similarity.

Results are summarized in the table below (see Evaluation section).

---

## Fine-Tuning with LoRA

###  Approach

To make training efficient on limited hardware, we fine-tuned **Qwen2-VL 2B** using **Low-Rank Adaptation (LoRA)** through the **Unsloth** library. Key benefits:
- Low GPU memory footprint with 4-bit quantization.
- Support for training both vision and language modules.

---

### Configuration Summary

| Parameter                | Value                                  |
|--------------------------|----------------------------------------|
| **Base Model**           | `unsloth/Qwen2-VL-2B-Instruct-bnb-4bit`|
| **Quantization**         | 4-bit                                  |
| **LoRA Rank (r)**        | 8 and 16 (both tested)                 |
| **Batch Size**           | 32                                     |
| **Epochs**               | 1 (initial testing)                    |
| **Learning Rate**        | 2e-4                                   |
| **Optimizer**            | AdamW (8-bit)                          |
| **Gradient Accumulation**| 1                                      |
| **Device**               | Google Colab GPU                       |

---

## Evaluation Metrics & Results

### Strategy

We evaluated both baseline and fine-tuned models using:

- **String Matching Accuracy**  
- **BERTScore (F1)**  
- **Wu-Palmer Similarity (WUP)**  

These metrics allowed us to capture both literal and semantic correctness of the predictions.

---

### Evaluation Results of Baseline vs Fine-Tuned Models

| Model                     | LoRA Rank | String Match ↑ | BERTScore (F1) ↑ | WUP Similarity ↑ |
|-------------------------- |-----------|----------------|------------------|------------------|
| Qwen-VL 2B (Baseline)     | N/A       | 10.32%         | 0.9288           | 0.4123           |
| Qwen-VL 7B (Baseline)     | N/A       | 27.10%         | 0.9122           | 0.4177           |
| Qwen-VL 2B (Fine-tuned)   | 4         | 47.28%         | 0.9598           | 0.5740           |
| Qwen-VL 2B (Fine-tuned)   | 16        | **48.28%**     | **0.9603**       | **0.5690**       |

---


## **Conclusion**

- LoRA-based fine-tuning significantly improved model performance.
- Even the compact **Qwen-VL 2B (LoRA Rank 16)** outperformed the larger **Qwen-VL 7B baseline**, showcasing the effectiveness of parameter-efficient adaptation.
- Semantic metrics like **BERTScore** and **WUP** highlighted the model's improved understanding beyond surface-level matches.

---


> For code, datasets, and inference scripts, please refer to the project repository.
