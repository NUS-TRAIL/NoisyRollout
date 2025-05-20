import os
import math
from PIL import Image
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from utils.model_parser import llm_eval_score
from mathruler.grader import extract_boxed_content, grade_answer

def load_image(image_path: str, min_pixels: int, max_pixels: int) -> Image.Image:
    """Load and preprocess an image"""
    try:
        image = Image.open(image_path).convert("RGB")
        
        # Resize if too large or too small
        if (image.width * image.height) > max_pixels:
            resize_factor = math.sqrt(max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))
        
        if (image.width * image.height) < min_pixels:
            resize_factor = math.sqrt(min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))
        
        return image
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

def prepare_prompts(dataset_name: str, samples: List[Dict], args) -> Tuple[List[Dict], List[Dict]]:
    """Prepare prompts for all samples"""
    prompts = []
    metadata = []
    
    for item in tqdm(samples, desc=f"Preparing {dataset_name} prompts"):
        # Skip if image doesn't exist
        if not os.path.exists(item["image_path"]):
            continue
        
        # Load image
        image = load_image(item["image_path"], args.min_pixels, args.max_pixels)
        if image is None:
            continue
        
        # Create prompt
        if args.version == "7b":
            prompt_text = f"<|im_start|>system\n{args.system_prompt}<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{item['question']}<|im_end|>\n<|im_start|>assistant\n"
        elif args.version == "32b":
            prompt_text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{args.system_prompt} {item['question']}<|im_end|>\n<|im_start|>assistant\n"
        else:
            raise
        
        prompts.append({
            "prompt": prompt_text,
            "multi_modal_data": {"image": image},
        })
        
        metadata.append({
            "dataset": dataset_name,
            "id": item["id"],
            "question": item["question"],
            "answer": item["answer"],
            "prompt": prompt_text,
            **{k: v for k, v in item.items() if k not in ["image_path", "dataset", "id", "question", "answer"]}
        })
    
    return prompts, metadata

def evaluate_prediction(prediction: str, answer: str, dataset: str, question: str = "") -> float:
    """Evaluate a prediction against the ground truth"""
    if dataset == "geo3k":
        extracted_answer = extract_boxed_content(prediction)
        return 1.0 if grade_answer(extracted_answer, answer) else 0.0
    
    elif dataset == "mathvista" or dataset == "mathverse" or dataset == "mathvision" or dataset == "wemath":
        try:
            score = llm_eval_score(question, prediction, answer, dataset)
        except:
            import time
            time.sleep(10)
            score = llm_eval_score(question, prediction, answer, dataset)
        return score
    
    if dataset == "hallubench":
        prediction = prediction.replace("\\boxed{}", "")
        extracted_answer = extract_boxed_content(prediction)
        return 1.0 if extracted_answer.lower() == answer else 0.0
        # return 1.0 if answer.lower() in prediction.lower() else 0.0
    
    else:
        # Default evaluation
        return 1.0 if extracted_answer == answer else 0.0

def process_outputs(outputs, metadata, max_workers: int) -> Dict[str, List[Dict]]:
    """Process model outputs and calculate metrics"""
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        for i, output in enumerate(outputs):
            prediction = output.outputs[0].text.strip()
            meta = metadata[i]
            dataset = meta["dataset"]
            if "question_for_eval" in meta:
                question = meta["question_for_eval"]
            else:
                question = meta["question"]
            
            future = executor.submit(
                evaluate_prediction, 
                prediction, 
                meta["answer"], 
                dataset,
                question
            )
            futures.append((future, i, prediction, meta))
        
        for future, i, prediction, meta in tqdm(futures, desc="Evaluating predictions"):
            try:
                accuracy = future.result()
                
                result = {
                    "id": meta["id"],
                    "question": meta["question"],
                    "answer": meta["answer"],
                    "prediction": prediction,
                    "accuracy": accuracy,
                    "correct": accuracy > 0,
                    **{k: v for k, v in meta.items() if k not in ["dataset", "id", "question", "answer"]}
                }
                
                results.append(result)
            except Exception as e:
                print(f"Error evaluating prediction {i}: {str(e)}")
    
    return results

def calculate_metrics(results: List[Dict]) -> Dict:
    """Calculate evaluation metrics"""
    if not results:
        return {"accuracy": 0.0}
    
    accuracy = sum(1 for r in results if r["correct"]) / len(results)
    metrics = {"accuracy": accuracy}
    
    # Calculate task-specific accuracies if available
    if any("task" in r for r in results):
        task_results = {}
        for r in results:
            if "task" in r:
                task = r["task"]
                if task not in task_results:
                    task_results[task] = []
                task_results[task].append(r["correct"])
        
        task_accuracies = {task: sum(results) / len(results) for task, results in task_results.items()}
        metrics["sub_accuracies"] = task_accuracies
    
    # Calculate problem version accuracies if available
    if any("problem_version" in r for r in results):
        version_results = {}
        for r in results:
            if "problem_version" in r:
                version = r["problem_version"]
                if version not in version_results:
                    version_results[version] = []
                version_results[version].append(r["correct"])
        
        version_accuracies = {version: sum(results) / len(results) for version, results in version_results.items()}
        metrics["sub_accuracies"] = version_accuracies
    
    # Calculate subject accuracies if available
    if any("subject" in r for r in results):
        subject_results = {}
        for r in results:
            if "subject" in r:
                subject = r["subject"]
                if subject not in subject_results:
                    subject_results[subject] = []
                subject_results[subject].append(r["correct"])
        
        subject_accuracies = {subject: sum(results) / len(results) for subject, results in subject_results.items()}
        metrics["sub_accuracies"] = subject_accuracies
    
    return metrics