import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch
from peft import PeftModel



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help='Path to image folder')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to image-metadata CSV')
    args = parser.parse_args()

    # Load metadata CSV
    df = pd.read_csv(args.csv_path)
    df = df[:10]  # Limit to first 10 rows for testing

    # Load model and processor, move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = FastVisionModel.from_pretrained(
    "drd01/qwen_for_abo_high_rank",
    load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
    )
    # model = PeftModel.from_pretrained(model, "SFT/model/qwen_qlora_16")
    FastVisionModel.for_inference(model)


    generated_answers = []
    for idx, row in tqdm(df.iterrows(), total=10):
        image_path = f"{args.image_dir}/{row['image_path']}"
        question = f"{str(row['question'])}. Answer with one word."
        try:
            image = Image.open(image_path).convert("RGB")
            image = image.resize((224, 224))
            messages = [
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]}
                ]
            input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
            inputs = tokenizer(
                    image,
                    input_text,
                    add_special_tokens = False,
                    return_tensors = "pt",
                ).to(device)
            generated_answer = model.generate(**inputs, max_length=300, do_sample=True, temperature=0.7)
    
            answer = tokenizer.decode(generated_answer[0], skip_special_tokens=True)
            


        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            answer = "error"
        # Ensure answer is one word and in English (basic post-processing)
        answer = str(answer).split('assistant\n')[-1].lower()
        print(f"Generated answer: {answer}")
        generated_answers.append(answer)
    # model.push_to_hub("drd01/qwen_for_abo_high_rank")
    # model.save_pretrained(f"SFT/model/qwen_qlora_4")  # Local saving
    # tokenizer.save_pretrained(f"SFT/model/qwen_qlora_4")
    df["generated_answer"] = generated_answers
    df.to_csv("results.csv", index=False)


if __name__ == "__main__":
    main()