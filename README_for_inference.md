# FastVisionModel Inference Script

This script performs visual question answering using the `unsloth` library's `FastVisionModel`, specifically using the `drd01/qwen_for_abo_high_rank` model file uploaded by us on HuggingFace.

It takes a folder of images and a CSV file containing image names and questions, runs inference to generate answers (as one-word responses), and outputs a new CSV (`results.csv`) with the predictions.

---

##  Prerequisites

Ensure that you have Python 3.10+ installed.

Install the required packages using the provided `requirements.txt` file:

```bash
cd inference_script
pip install -r requirements.txt
```

---

## Folder Structure

```
project_root/
│
├── inference_script/
│   ├── inference.py              # Your main script
│   ├── requirements.txt          # Dependencies
│
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...                      
│
├── questions.csv                 # CSV with image names and questions
└── ...
```

---

## Input CSV Format

The CSV (`questions.csv`) should have the following format:

```csv
image_name,question
image1.jpg,What is the color of the object?
image2.jpg,What type of furniture is this?
...
```

---

##  How to Run

Run the script using:

```bash
python inference_script/inference.py \
  --image_dir path/to/images \
  --csv_path path/to/questions.csv
```

Example:

```bash
python inference_script/inference.py \
  --image_dir ./images \
  --csv_path ./questions.csv
```

---

## Output

The script will generate a `results.csv` file with an additional column:

```csv
image_name,question,generated_answer
image1.jpg,What is the color of the object?,red
image2.jpg,What type of furniture is this?,chair
...
```

---

##  Notes

- Images are resized to 224x224 before inference.
- The script automatically uses GPU (CUDA) if available.
- It generates **one-word answers** with simple post-processing.
- Errors during image processing will result in `"error"` as the answer in the CSV.

---

##  requirements.txt

Make sure your `requirements.txt` file inside `inference_script/` contains:

```txt
torch
unsloth
pillow
pandas
tqdm
```

---

##  Contact

For any issues or questions, please contact anukriti.singh@iiitb.ac.in or mohammed.danishrabbani@iiitb.ac.in
