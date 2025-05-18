# Multimodal Visual Question Answering using the ABO Dataset

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
  - `VQA-dataset-train/merged_listings_train.csv`
  - `VQA-dataset-test/merged_listings_test.csv`

---

## Baseline Evaluation

### Objective

To benchmark performance, we evaluated off-the-shelf VQA models on our curated dataset **without any fine-tuning**.

### Models Used

| Model         | Description                                                   |
|---------------|---------------------------------------------------------------|
| **Qwen-VL 7B** | Multimodal transformer with high vision-language capability.  |
| **Qwen-VL 2B** | Smaller variant optimized for low-latency inference.          |
| **BLIP Base**  | BLIP model trained for general VQA and captioning tasks.     |

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
|--------------------------|-----------|----------------|------------------|------------------|
| Qwen-VL 2B (Baseline)     | N/A       | 31.5%          | 0.839            | 0.752            |
| Qwen-VL 7B (Baseline)     | N/A       | 38.2%          | 0.874            | 0.791            |
| BLIP Base (Baseline)      | N/A       | 27.6%          | 0.812            | 0.701            |
| Qwen-VL 2B (Fine-tuned)   | **8**     | **43.4%**      | **0.897**        | **0.824**        |
| Qwen-VL 2B (Fine-tuned)   | **16**    | **45.1%**      | **0.911**        | **0.839**        |

---
## Project Directory Structure

├── Evaluation
│ ├── evaluation_BaselineModels.ipynb
│ └── evaluation_FinetunedModel.ipynb
├── Inference
│ ├── inference_baseline_QWEN2B.ipynb
│ └── inference_baseline_QWEN7B.ipynb
├── README.md
├── Results
│ ├── results_QWEN7B.csv
│ ├── results_QWEN_finetuned_rank16.csv
│ ├── results_QWEN_finetuned_rank4.csv
│ └── results_qwen2B.csv
├── SFT
│ ├── model
│ │ └── qwen_qlora_4
│ │ └── [...model files...]
│ ├── qwen2-5-vl-2b-vision.ipynb
│ ├── qwen_2.5_low_rank_checkpoints
│ │ └── [...checkpoints...]
│ ├── qwen_training.py
│ ├── qwen_training_r_16.py
│ └── qwen_training_r_8.py
├── dataset
│ ├── VQA-dataset
│ │ └── [...listing CSVs...]
│ ├── VQA-dataset-test
│ │ └── [...test CSVs...]
│ ├── VQA-dataset-train
│ │ └── [...train CSVs...]
│ └── [...data curation notebooks...]
├── inference.py
└── results.csv

## Conclusion

- LoRA-based fine-tuning significantly improved model performance.
- Even the compact **Qwen-VL 2B (LoRA Rank 16)** outperformed the larger **Qwen-VL 7B baseline**, showcasing the effectiveness of parameter-efficient adaptation.
- Semantic metrics like **BERTScore** and **WUP** highlighted the model's improved understanding beyond surface-level matches.

---

## Future Directions

- Add **multilingual question generation** using ABO’s multilingual metadata.
- Integrate **ranking-based loss functions** for improved discriminative training.
- Deploy the fine-tuned model for real-time product Q&A in e-commerce settings.

---

> For code, datasets, and inference scripts, please refer to the project repository.
