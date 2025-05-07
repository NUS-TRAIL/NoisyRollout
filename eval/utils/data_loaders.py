import os
import json
import pandas as pd
from PIL import Image
from tqdm import tqdm
from typing import List, Dict
from datasets import load_dataset

def load_geo3k_dataset(data_path: str) -> List[Dict]:
    """Load Geo3K dataset"""
    data_path = os.path.join(data_path, "geometry3k/test")
    dataset = []
    folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    
    for folder in tqdm(folders, desc="Loading Geo3K data"):
        folder_path = os.path.join(data_path, folder)
        image_path = os.path.join(folder_path, "img_diagram.png")
        json_path = os.path.join(folder_path, "data.json")
        
        if not os.path.exists(image_path) or not os.path.exists(json_path):
            continue
        
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        
        dataset.append({
            "id": data["id"],
            "image_path": image_path,
            "question": data["annotat_text"],
            "answer": data["choices"][mapping[data["answer"]]],
            "dataset": "geo3k"
        })
    
    return dataset

def load_wemath_dataset(data_path: str) -> List[Dict]:
    """Load WeMath dataset"""
    image_root = os.path.join(data_path, "wemath/images")
    data_path = os.path.join(data_path, "wemath/testmini.json")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    dataset = []
    for item in data:
        # Determine the image path
        image_path = os.path.join(image_root, item["image_path"])

        dataset.append({
            "id": item["ID"] + "@" + item["key"],
            "image_path": image_path,
            "question": f"{item['question']}\n\nOptions: {item['option']}",
            "answer": item["answer"],
            "dataset": "wemath"
        })
    
    return dataset

def load_mathvista_dataset(data_path: str) -> List[Dict]:
    """Load MathVista dataset"""
    image_base_dir = os.path.join(data_path, "mathvista")
    dataset_raw = load_dataset("AI4Math/MathVista", split="testmini")
    
    dataset = []
    mapping = {
        "0": "A", "1": "B", "2": "C", "3": "D",
        "4": "E", "5": "F", "6": "G", "7": "H"
    }
    
    for item in dataset_raw:
        if item["question_type"] == "multi_choice":
            idx = item["choices"].index(item["answer"])
            answer = mapping[str(idx)]
        else:
            answer = item["answer"]
        
        dataset.append({
            "id": item.get("pid", ""),
            "image_path": os.path.join(image_base_dir, item["image"]),
            "question": item["query"],
            "answer": answer,
            "task": item["metadata"]["task"],
            "dataset": "mathvista"
        })
    
    return dataset

def load_mathverse_dataset(data_path: str) -> List[Dict]:
    """Load MathVerse dataset"""
    image_base_dir = os.path.join(data_path, "mathverse/images")
    data_path = os.path.join(data_path, "mathverse/testmini.json")
    
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    dataset = []
    for item in data:
        dataset.append({
            "id": item.get("sample_index", ""),
            "image_path": os.path.join(image_base_dir, item["image"]),
            "question": item["query_cot"],
            "question_for_eval": item["question_for_eval"],
            "answer": item["answer"],
            "problem_version": item["problem_version"],
            "dataset": "mathverse"
        })
    
    return dataset

def load_mathvision_dataset(data_path: str) -> List[Dict]:
    """Load MathVision dataset"""
    image_base_dir = os.path.join(data_path, "mathvision/images")
    data_path = os.path.join(data_path, "mathvision/MathVision.tsv")
    df = pd.read_csv(data_path, sep='\t')
    
    dataset = []
    for _, row in df.iterrows():
        dataset.append({
            "id": row.get("index", ""),
            "image_path": os.path.join(image_base_dir, f"{row['index']}.jpg"),
            "question": row["question"],
            "answer": row["answer"],
            "subject": row.get("category", "unknown"),
            "dataset": "mathvision"
        })
    
    return dataset

def load_hallubench_dataset(data_path: str) -> List[Dict]:
    """Load Hallubench dataset"""
    image_base_dir = os.path.join(data_path, "hallubench/images")
    data_path = os.path.join(data_path, "hallubench/HallusionBench.json")
    
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    dataset = []
    for item in data:
        if not item["filename"]:
            continue
        
        if "?" in item["question"]:
            question = item["question"].split("?")[:-1][0]
        else:
            question = item["question"]
        question += "? You final answer can only be \\boxed{yes} or \\boxed{no}."
        gt_answer = "yes" if int(item["gt_answer"]) == 1 else "no"
        sid, fid, qid = item["set_id"], item["figure_id"], item["question_id"]
        dataset.append({
            "id": f"{sid}_{fid}_{qid}",
            "image_path": os.path.join(image_base_dir, item["filename"].replace("./", "")),
            "question": question,
            "question_for_eval": question,
            "answer": gt_answer,
            "problem_version": item["subcategory"],
            "dataset": "hallubench"
        })
    
    return dataset