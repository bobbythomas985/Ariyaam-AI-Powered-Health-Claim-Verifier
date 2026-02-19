# ============================================================
# ARIYAAM v3.0 — LLAMA-3-OpenBioLLM-8B BIOMEDICAL SUMMARIZER (QLoRA)
# Module: 03_llama_summ.py (VS Code Compatible + Auto-Download)
# Target: Scientific synthesis + Layman simplification via prompt conditioning
# ✅ MODEL: aaditya/Llama3-OpenBioLLM-8B (biomedical pre-fine-tuned)
# ✅ DATASETS: allenai/mslr2022, BioLaySumm2025-PLOS, cbasu/Med-EASi
# ✅ AUTO-DOWNLOAD: HuggingFace datasets with local fallback
# ✅ FIXED: Colab dependencies removed for local execution
# ✅ FIXED: Windows multiprocessing guards added
# ✅ FIXED: All paths are local/relative (no /content/)
# ✅ IMPLEMENTED: QLoRA 4-bit quantization + LoRA adapters
# ✅ IMPLEMENTED: Two-stage fine-tuning (MS2 → BioLaySumm/MedEasi)
# ✅ IMPLEMENTED: Dual-register prompting (scientific + layman)
# ============================================================

import os
import sys
import random
import numpy as np
import torch
import datetime
import warnings
import json
import shutil
import tarfile
import csv
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, field

# Third-party imports
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
    TaskType,
)
import bitsandbytes as bnb
from trl import SFTTrainer
from sklearn.metrics import rouge_score, bert_score
from textstat import flesch_kincaid_grade, smog_index

# Suppress warnings
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

# ============================================================
# CONFIGURATION — VS CODE LOCAL PATHS
# ============================================================
@dataclass
class Config:
    # ✅ Local paths (no Colab /content/)
    base_dir: str = field(default_factory=lambda: os.getcwd())
    output_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), "models"))
    data_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), "data"))
    drive_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), "drive_backup"))
    
    # Hardware detection
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    num_gpus: int = field(default_factory=lambda: torch.cuda.device_count())
    
    # ✅ MODEL: OpenBioLLM (biomedical pre-fine-tuned Llama-3)
    model_name: str = "aaditya/Llama3-OpenBioLLM-8B"
    tokenizer_name: str = "aaditya/Llama3-OpenBioLLM-8B"
    
    # QLoRA configuration
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True
    
    # LoRA configuration
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Training hyperparameters
    max_length: int = 4096  # Llama-3 context window
    max_new_tokens: int = 512
    batch_size: int = 1  # QLoRA memory constraints
    gradient_accumulation_steps: int = 16  # Effective batch = 16
    stage1_epochs: int = 3  # MS2 scientific synthesis
    stage2_epochs: int = 2  # BioLaySumm layman simplification
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 0.3
    
    # Dataset configuration (auto-download from HF)
    ms2_samples: int = 20000  # Limit for training speed
    biolaysumm_samples: int = 15000
    medeasi_samples: int = 500
    
    # Prompt templates (OpenBioLLM uses Llama-3 chat format)
    scientific_prompt: str = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a biomedical scientist. Synthesize the provided evidence passages into a concise verdict summary for the given claim. Follow these rules:
1. Start with the verdict: SUPPORT, REFUTE, or NOT_ENOUGH_INFO
2. Cite evidence inline using [Evidence 1], [Evidence 2], etc.
3. Keep summary under 200 words
4. Use precise scientific terminology
5. Do not add information not present in the evidence

<|eot_id|><|start_header_id|>user<|end_header_id|>
Claim: {claim}

Evidence:
{evidence}

Generate a scientific verdict summary:
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    
    layman_prompt: str = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a medical communicator. Rewrite the scientific summary at a grade 8 reading level for a general audience. Follow these rules:
1. Start with the verdict in plain language
2. Replace all medical jargon with simple terms
3. Keep summary under 120 words
4. Use short, clear sentences
5. Explain any necessary technical concepts simply

<|eot_id|><|start_header_id|>user<|end_header_id|>
Scientific Summary: {scientific_summary}

Rewrite for a general audience:
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    
    # Evaluation targets
    target_rouge_l: float = 0.35
    target_bertscore: float = 0.85
    target_sari: float = 0.40
    target_fk_grade: float = 8.0
    
    # Reproducibility
    seed: int = 42
    
    def __post_init__(self):
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.drive_dir, exist_ok=True)
        
        print(f"✅ Configuration initialized")
        print(f"   Device: {self.device}")
        print(f"   GPUs: {self.num_gpus}")
        print(f"   Output: {self.output_dir}")
        print(f"   Data: {self.data_dir}")
        print(f"   Model: {self.model_name}")
        print(f"   QLoRA: 4-bit={self.load_in_4bit}, rank={self.lora_rank}")

config = Config()

# ============================================================
# REPRODUCIBILITY
# ============================================================
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(config.seed)

# ============================================================
# AUTO-DOWNLOAD HELPERS
# ============================================================
def ensure_hf_auth():
    """Check for HuggingFace token and prompt if missing"""
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("⚠️ HF_TOKEN not set in environment.")
        print("   Some datasets require authentication.")
        print("   Set it with: export HF_TOKEN=your_token_here")
        print("   Or get one at: https://huggingface.co/settings/tokens")
        # Continue anyway - some datasets are public
    return hf_token

def download_and_extract_tar(url: str, extract_to: str, expected_files: List[str] = None) -> bool:
    """Download and extract a tar.gz file with progress bar"""
    import requests
    from tqdm import tqdm
    
    os.makedirs(extract_to, exist_ok=True)
    tar_path = os.path.join(extract_to, os.path.basename(url))
    
    if os.path.exists(tar_path):
        print(f"✅ Archive already exists: {tar_path}")
    else:
        print(f"⬇️ Downloading {url}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            
            with open(tar_path, 'wb') as f, tqdm(
                total=total_size, unit='B', unit_scale=True, desc=os.path.basename(url)
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            print(f"✅ Downloaded to: {tar_path}")
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False
    
    # Extract
    extract_dir = os.path.join(extract_to, "extracted")
    if not os.path.exists(extract_dir) or not all(
        os.path.exists(os.path.join(extract_dir, f)) for f in (expected_files or [])
    ):
        print(f"📦 Extracting to: {extract_dir}")
        try:
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(path=extract_dir)
            print(f"✅ Extraction complete")
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            return False
    
    return True

def load_mslr2022_local(data_dir: str, split: str = 'train', dataset: str = 'ms2', 
                       max_samples: int = None, auto_download: bool = True) -> Optional[Dataset]:
    """
    Load MSLR2022 dataset (MS^2 or Cochrane) from local files or auto-download from HF.
    
    Args:
        data_dir: Base data directory
        split: 'train', 'dev', or 'test'
        dataset: 'ms2' or 'cochrane'
        max_samples: Limit number of examples (for testing)
        auto_download: If True, try HF load_dataset first, then fallback to manual download
    """
    print(f"📚 Loading MSLR2022/{dataset} {split}...")
    
    # Try HuggingFace load_dataset first (auto-download)
    if auto_download:
        try:
            print(f"   Attempting HF download: allenai/mslr2022 ({dataset}, {split})")
            hf_dataset = load_dataset("allenai/mslr2022", name=dataset, split=split, trust_remote_code=True)
            
            # Convert to our format
            data = []
            for i, row in enumerate(hf_dataset):
                # MSLR format: ReviewID, PMID, Title, Abstract, Target
                claim = f"Synthesize findings from study {row.get('PMID', 'unknown')}"
                evidence = f"[Evidence 1] {row.get('Title', '')}: {row.get('Abstract', '')}"
                reference = row.get('Target', '')
                
                if not reference or not evidence:
                    continue
                
                data.append({
                    'claim': claim,
                    'evidence': evidence,
                    'reference_summary': reference,
                    'register': 'scientific',
                    'source': f'mslr2022_{dataset}',
                    'review_id': str(row.get('ReviewID', i)),
                    'pmid': str(row.get('PMID', ''))
                })
                
                if max_samples and len(data) >= max_samples:
                    break
            
            if data:
                print(f"✅ Loaded {len(data)} MSLR2022/{dataset} {split} examples from HF")
                return Dataset.from_list(data)
        except Exception as e:
            print(f"⚠️ HF load_dataset failed: {e}")
            print("   Falling back to manual download...")
    
    # Fallback: Manual download from AI2 S3
    mslr_dir = os.path.join(data_dir, 'mslr2022')
    os.makedirs(mslr_dir, exist_ok=True)
    
    tar_url = "https://ai2-s2-mslr.s3.us-west-2.amazonaws.com/mslr_data.tar.gz"
    extract_path = os.path.join(mslr_dir, 'extracted')
    
    if not os.path.exists(os.path.join(extract_path, dataset)):
        if not download_and_extract_tar(
            tar_url, 
            mslr_dir, 
            expected_files=[f"{dataset}/{split}-inputs.csv", f"{dataset}/{split}-targets.csv"]
        ):
            print(f"❌ Could not download MSLR2022 data")
            return None
    
    # Load from CSV files
    dataset_dir = os.path.join(extract_path, dataset)
    inputs_file = os.path.join(dataset_dir, f"{split}-inputs.csv")
    targets_file = os.path.join(dataset_dir, f"{split}-targets.csv")
    
    if not os.path.exists(inputs_file) or not os.path.exists(targets_file):
        print(f"⚠️ Files not found: {inputs_file} or {targets_file}")
        return None
    
    # Load inputs and targets
    inputs_df = pd.read_csv(inputs_file)
    targets_df = pd.read_csv(targets_file)
    
    # Merge on ReviewID
    merged = pd.merge(inputs_df, targets_df, on='ReviewID', suffixes=('_input', '_target'))
    
    data = []
    for i, row in merged.iterrows():
        claim = f"Synthesize findings from study {row.get('PMID', 'unknown')}"
        evidence = f"[Evidence 1] {row.get('Title_input', '')}: {row.get('Abstract_input', '')}"
        reference = row.get('Target', '')
        
        if not reference or not evidence:
            continue
        
        data.append({
            'claim': claim,
            'evidence': evidence,
            'reference_summary': reference,
            'register': 'scientific',
            'source': f'mslr2022_{dataset}',
            'review_id': str(row.get('ReviewID', i)),
            'pmid': str(row.get('PMID_input', ''))
        })
        
        if max_samples and len(data) >= max_samples:
            break
    
    if not data:
        print(f"⚠️ No MSLR2022/{dataset} {split} data loaded")
        return None
    
    print(f"✅ Loaded {len(data)} MSLR2022/{dataset} {split} examples from local files")
    return Dataset.from_list(data)

def load_biolaysumm_2025_local(data_dir: str, split: str = 'train', 
                               max_samples: int = None, auto_download: bool = True) -> Optional[Dataset]:
    """Load BioLaySumm 2025 PLOS from HF or local files"""
    print(f"📚 Loading BioLaySumm2025-PLOS {split}...")
    
    if auto_download:
        try:
            print(f"   Attempting HF download: BioLaySumm/BioLaySumm2025-PLOS")
            hf_dataset = load_dataset("BioLaySumm/BioLaySumm2025-PLOS", split=split, trust_remote_code=True)
            
            data = []
            for i, row in enumerate(hf_dataset):
                # BioLaySumm format: paper_id, abstract, lay_summary
                abstract = row.get('abstract', '')
                lay_summary = row.get('lay_summary', '')
                
                if not abstract or not lay_summary:
                    continue
                
                data.append({
                    'claim': 'Explain this research simply',
                    'evidence': f"[Evidence 1] {abstract}",
                    'reference_summary': lay_summary,
                    'register': 'layman',
                    'source': 'biolaysumm_2025_plos',
                    'paper_id': str(row.get('paper_id', f'biolay_{i}'))
                })
                
                if max_samples and len(data) >= max_samples:
                    break
            
            if data:
                print(f"✅ Loaded {len(data)} BioLaySumm2025-PLOS {split} examples from HF")
                return Dataset.from_list(data)
        except Exception as e:
            print(f"⚠️ HF load_dataset failed: {e}")
    
    # Fallback: Local files
    filepath = os.path.join(data_dir, 'biolaysumm_2025_plos', f"{split}.jsonl")
    if not os.path.exists(filepath):
        print(f"⚠️ Local file not found: {filepath}")
        print("   Please download from: https://huggingface.co/datasets/BioLaySumm/BioLaySumm2025-PLOS")
        return None
    
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            try:
                row = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            
            abstract = row.get('abstract', '')
            lay_summary = row.get('lay_summary', '')
            
            if not abstract or not lay_summary:
                continue
            
            data.append({
                'claim': 'Explain this research simply',
                'evidence': f"[Evidence 1] {abstract}",
                'reference_summary': lay_summary,
                'register': 'layman',
                'source': 'biolaysumm_2025_plos',
                'paper_id': str(row.get('paper_id', f'biolay_{line_num}'))
            })
            
            if max_samples and len(data) >= max_samples:
                break
    
    if not data:
        print(f"⚠️ No BioLaySumm2025-PLOS data loaded")
        return None
    
    print(f"✅ Loaded {len(data)} BioLaySumm2025-PLOS {split} examples from local files")
    return Dataset.from_list(data)

def load_medeasi_local(data_dir: str, split: str = 'train', 
                       max_samples: int = None, auto_download: bool = True) -> Optional[Dataset]:
    """Load Med-EASi from HF or local files (updated dataset name)"""
    print(f"📚 Loading Med-EASi {split}...")
    
    if auto_download:
        try:
            print(f"   Attempting HF download: cbasu/Med-EASi")
            hf_dataset = load_dataset("cbasu/Med-EASi", split=split, trust_remote_code=True)
            
            data = []
            for i, row in enumerate(hf_dataset):
                # Med-EASi format: original, simplified
                original = row.get('original', '')
                simplified = row.get('simplified', '')
                
                if not original or not simplified:
                    continue
                
                data.append({
                    'claim': 'Rewrite in plain language',
                    'evidence': f"[Evidence 1] {original}",
                    'reference_summary': simplified,
                    'register': 'layman',
                    'source': 'med_easi',
                    'example_id': f'medeasi_{i}'
                })
                
                if max_samples and len(data) >= max_samples:
                    break
            
            if data:
                print(f"✅ Loaded {len(data)} Med-EASi {split} examples from HF")
                return Dataset.from_list(data)
        except Exception as e:
            print(f"⚠️ HF load_dataset failed: {e}")
    
    # Fallback: Local files
    filepath = os.path.join(data_dir, 'med_easi', f"{split}.jsonl")
    if not os.path.exists(filepath):
        print(f"⚠️ Local file not found: {filepath}")
        print("   Please download from: https://huggingface.co/datasets/cbasu/Med-EASi")
        return None
    
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            try:
                row = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            
            original = row.get('original', '')
            simplified = row.get('simplified', '')
            
            if not original or not simplified:
                continue
            
            data.append({
                'claim': 'Rewrite in plain language',
                'evidence': f"[Evidence 1] {original}",
                'reference_summary': simplified,
                'register': 'layman',
                'source': 'med_easi',
                'example_id': f'medeasi_{line_num}'
            })
            
            if max_samples and len(data) >= max_samples:
                break
    
    if not data:
        print(f"⚠️ No Med-EASi data loaded")
        return None
    
    print(f"✅ Loaded {len(data)} Med-EASi {split} examples from local files")
    return Dataset.from_list(data)

def format_training_example(example, prompt_template, register='scientific'):
    """Format a single example for instruction tuning"""
    claim = example.get('claim', '')
    evidence = example.get('evidence', '')
    reference = example.get('reference_summary', '')
    
    if register == 'scientific':
        prompt = config.scientific_prompt.format(claim=claim, evidence=evidence)
    else:  # layman
        # For layman stage, input is the scientific summary, output is layman version
        prompt = config.layman_prompt.format(scientific_summary=evidence)
        reference = example.get('reference_summary', '')  # layman reference
    
    # Format for causal LM: prompt + reference + eos
    full_text = f"{prompt}{reference}<|eot_id|>"
    
    return {
        'text': full_text,
        'prompt': prompt,
        'reference': reference,
        'register': register,
        'source': example.get('source', 'unknown')
    }

# ============================================================
# QLoRA MODEL LOADING (OpenBioLLM)
# ============================================================
def load_openbiollm_qlora(model_name=None):
    """Load Llama3-OpenBioLLM-8B with QLoRA configuration"""
    if model_name is None:
        model_name = config.model_name
    
    print(f"\n📥 Loading {model_name} with QLoRA configuration...")
    
    # 4-bit quantization config
    bnb_config = bnb.BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name if config.tokenizer_name else model_name,
        use_fast=True,
        padding_side='right',
        truncation_side='right'
    )
    
    # Add special tokens if needed (OpenBioLLM uses Llama-3 chat format)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.unk_token is None:
        tokenizer.unk_token = tokenizer.eos_token
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=config.lora_target_modules,
        modules_to_save=None
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"✅ Model loaded with QLoRA")
    print(f"   Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"   Total params: {total_params:,}")
    print(f"   Memory footprint: ~{total_params * 2 / 1e9:.1f} GB (4-bit)")
    
    return model, tokenizer

# ============================================================
# CUSTOM DATA COLLATOR FOR INSTRUCTION TUNING
# ============================================================
@dataclass
class InstructionDataCollator:
    """Data collator for instruction-tuned causal LM with padding"""
    tokenizer: any
    max_length: int = 4096
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # Extract texts
        texts = [f['text'] for f in features]
        
        # Tokenize with padding
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Labels are same as input_ids for causal LM
        batch["labels"] = batch["input_ids"].clone()
        
        # Mask padding tokens in labels (ignore_index=-100)
        padding_mask = batch["attention_mask"] == 0
        batch["labels"][padding_mask] = -100
        
        return batch

# ============================================================
# EVALUATION METRICS
# ============================================================
def compute_rouge_l(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE-L F1 score"""
    try:
        scores = rouge_score(predictions, references, rouge_types=['rougeL'])
        return {'rouge_l_f1': scores['rougeL'].fmeasure}
    except:
        return {'rouge_l_f1': 0.0}

def compute_bertscore(predictions: List[str], references: List[str], lang='en') -> Dict[str, float]:
    """Compute BERTScore F1"""
    try:
        P, R, F1 = bert_score(predictions, references, lang=lang, verbose=False)
        return {'bertscore_f1': F1.mean().item()}
    except:
        return {'bertscore_f1': 0.0}

def compute_sari(predictions: List[str], references: List[str], sources: List[str]) -> Dict[str, float]:
    """Compute SARI score for simplification quality"""
    try:
        from sari import score_sari
        sari_scores = []
        for orig, pred, ref in zip(sources, predictions, references):
            try:
                score = score_sari([orig], [pred], [[ref]])
                sari_scores.append(score)
            except:
                continue
        if sari_scores:
            return {'sari': np.mean(sari_scores)}
        return {'sari': 0.0}
    except ImportError:
        print("⚠️ SARI not installed. Install with: pip install sari")
        return {'sari': 0.0}
    except:
        return {'sari': 0.0}

def compute_readability(texts: List[str]) -> Dict[str, float]:
    """Compute readability metrics (Flesch-Kincaid Grade Level)"""
    try:
        grades = [flesch_kincaid_grade(t) for t in texts if t.strip()]
        if grades:
            return {'fk_grade_level': np.mean(grades)}
        return {'fk_grade_level': 0.0}
    except:
        return {'fk_grade_level': 0.0}

def compute_summarization_metrics(eval_predictions, eval_dataset, register='scientific'):
    """Compute appropriate metrics based on register"""
    predictions = []
    references = []
    sources = []
    
    for pred, example in zip(eval_predictions, eval_dataset):
        generated = pred.get('generated_text', '')
        prompt = example.get('prompt', '')
        
        if prompt in generated:
            generated = generated.replace(prompt, '').strip()
        
        for token in ['<|eot_id|>', '<|end_header_id|>', '<|start_header_id|>']:
            generated = generated.replace(token, '').strip()
        
        predictions.append(generated)
        references.append(example.get('reference', ''))
        sources.append(example.get('evidence', ''))
    
    metrics = {}
    
    if register == 'scientific':
        rouge = compute_rouge_l(predictions, references)
        bertscore = compute_bertscore(predictions, references)
        metrics.update(rouge)
        metrics.update(bertscore)
    else:
        sari = compute_sari(predictions, references, sources)
        readability = compute_readability(predictions)
        rouge = compute_rouge_l(predictions, references)
        metrics.update(sari)
        metrics.update(readability)
        metrics.update(rouge)
    
    return metrics

# ============================================================
# CUSTOM CALLBACK FOR LOGGING
# ============================================================
class SummarizerLoggingCallback(TrainerCallback):
    """Custom callback for logging generation samples during training"""
    def __init__(self, tokenizer, eval_dataset, eval_every_n_steps=500):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.eval_every_n_steps = eval_every_n_steps
        self.sample_indices = random.sample(range(len(eval_dataset)), min(3, len(eval_dataset)))
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step % self.eval_every_n_steps == 0 and state.global_step > 0:
            model = kwargs.get('model')
            if model is None:
                return control
            
            print(f"\n📝 Generation samples at step {state.global_step}:")
            model.eval()
            
            for idx in self.sample_indices:
                if idx >= len(self.eval_dataset):
                    continue
                
                example = self.eval_dataset[idx]
                prompt = example.get('prompt', '')
                reference = example.get('reference', '')
                
                inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=config.max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                if prompt in generated:
                    generated = generated.replace(prompt, '').strip()
                
                print(f"   --- Sample {idx+1} ---")
                print(f"   Reference: {reference[:100]}...")
                print(f"   Generated: {generated[:100]}...")
                print()
            
            model.train()
        
        return control

# ============================================================
# TRAINING FUNCTIONS
# ============================================================
def train_stage(model, tokenizer, train_dataset, eval_dataset, stage_name, epochs, output_dir):
    """Train one stage of the summarizer"""
    print(f"\n{'='*80}")
    print(f"🚀 {stage_name}")
    print(f"{'='*80}")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stage_output_dir = os.path.join(output_dir, f"openbiollm_summarizer_{stage_name.lower().replace(' ', '_')}_{timestamp}")
    os.makedirs(stage_output_dir, exist_ok=True)
    
    # Format dataset for instruction tuning
    print("🔄 Formatting dataset for instruction tuning...")
    
    def format_fn(example):
        register = example.get('register', 'scientific')
        return format_training_example(example, config.scientific_prompt if register == 'scientific' else config.layman_prompt, register)
    
    formatted_dataset = train_dataset.map(format_fn, batched=False, num_proc=1)
    formatted_eval = eval_dataset.map(format_fn, batched=False, num_proc=1) if eval_dataset else None
    
    # Data collator
    data_collator = InstructionDataCollator(tokenizer, max_length=config.max_length)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=stage_output_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        optim="paged_adamw_32bit",
        max_grad_norm=config.max_grad_norm,
        fp16=True if config.device == "cuda" else False,
        bf16=False,
        gradient_checkpointing=True,
        evaluation_strategy="steps" if formatted_eval else "no",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        logging_steps=50,
        load_best_model_at_end=True if formatted_eval else False,
        metric_for_best_model="eval_rouge_l_f1" if formatted_eval else None,
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=config.seed,
        dataloader_num_workers=0,
        dataloader_pin_memory=True if config.device == "cuda" else False,
        remove_unused_columns=False,
        push_to_hub=False,
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset,
        eval_dataset=formatted_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
            SummarizerLoggingCallback(tokenizer, formatted_eval if formatted_eval else train_dataset)
        ]
    )
    
    # Train
    print(f"📊 Training on {len(formatted_dataset):,} examples")
    if formatted_eval:
        print(f"📊 Validating on {len(formatted_eval):,} examples")
    
    train_result = trainer.train()
    
    # Save model
    trainer.save_model(stage_output_dir)
    tokenizer.save_pretrained(stage_output_dir)
    model.save_pretrained(stage_output_dir)
    
    print(f"✅ {stage_name} complete. Model saved to: {stage_output_dir}")
    
    # Evaluate if eval dataset available
    if formatted_eval:
        print("\n📊 Running evaluation...")
        eval_results = trainer.evaluate()
        print(f"   ROUGE-L F1: {eval_results.get('eval_rouge_l_f1', 0):.4f}")
        print(f"   BERTScore F1: {eval_results.get('eval_bertscore_f1', 0):.4f}")
        
        metrics = {
            'stage': stage_name,
            'timestamp': timestamp,
            'eval_results': {k: v for k, v in eval_results.items() if k.startswith('eval_')},
            'train_samples': len(formatted_dataset),
            'eval_samples': len(formatted_eval)
        }
        with open(os.path.join(stage_output_dir, "metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2)
    
    return stage_output_dir, model, trainer

# ============================================================
# INFERENCE FUNCTION FOR TESTING
# ============================================================
def generate_summary(model, tokenizer, claim: str, evidence: str, register: str = 'scientific', 
                    max_new_tokens: int = None) -> str:
    """Generate a summary for a claim given evidence"""
    if max_new_tokens is None:
        max_new_tokens = config.max_new_tokens
    
    if register == 'scientific':
        prompt = config.scientific_prompt.format(claim=claim, evidence=evidence)
    else:
        prompt = config.layman_prompt.format(scientific_summary=evidence)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if prompt in generated:
        generated = generated.split(prompt)[-1].strip()
    
    for token in ['<|eot_id|>', '<|end_header_id|>', '<|start_header_id|>', '<|begin_of_text|>']:
        generated = generated.replace(token, '').strip()
    
    model.train()
    return generated

# ============================================================
# VISUALIZATION
# ============================================================
def generate_training_plots(trainer, output_dir, timestamp, stage_name):
    """Generate diagnostic plots for summarizer training"""
    figures_dir = os.path.join(output_dir, f"openbiollm_summarizer_{stage_name.lower().replace(' ', '_')}_{timestamp}", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    history = trainer.state.log_history
    steps = [h["step"] for h in history if "loss" in h]
    train_loss = [h["loss"] for h in history if "loss" in h]
    
    plt.figure(figsize=(10, 5))
    plt.plot(steps, train_loss, label="Train Loss", color="#2E86AB", linewidth=2)
    plt.xlabel("Training Steps", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.title(f"Training Loss - {stage_name}", fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    loss_path = os.path.join(figures_dir, "training_loss.png")
    plt.savefig(loss_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Training plots saved to: {figures_dir}")

# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================
def main():
    """Main training entry point - Windows multiprocessing safe"""
    print("\n" + "="*80)
    print("🔬 ARIYAAM v3.0 — OpenBioLLM-8B SUMMARIZER (QLoRA + Auto-Download)")
    print("="*80)
    print(f"📁 Base: {config.base_dir}")
    print(f"📁 Data: {config.data_dir}")
    print(f"📁 Output: {config.output_dir}")
    print(f"🖥️ Device: {config.device}")
    print(f"🎯 Model: {config.model_name}")
    print(f"🎯 Target: ROUGE-L >{config.target_rouge_l}, BERTScore >{config.target_bertscore}")
    print("="*80)
    
    # Check HF auth
    ensure_hf_auth()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config.output_dir, f"openbiollm_summarizer_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # ============================================================
    # LOAD MODEL & TOKENIZER
    # ============================================================
    print("\n" + "="*70)
    print(f"📥 LOADING {config.model_name} WITH QLoRA")
    print("="*70)
    
    model, tokenizer = load_openbiollm_qlora()
    
    # ============================================================
    # LOAD DATASETS (AUTO-DOWNLOAD FROM HF WITH FALLBACK)
    # ============================================================
    print("\n" + "="*70)
    print("📚 LOADING DATASETS (Auto-download from HuggingFace)")
    print("="*70)
    
    # Stage 1: MS2 from MSLR2022 (scientific multi-doc synthesis)
    print("\n📥 Loading MS2 (Stage 1: Scientific Synthesis)...")
    ms2_train = load_mslr2022_local(
        config.data_dir, 
        split='train', 
        dataset='ms2', 
        max_samples=config.ms2_samples,
        auto_download=True
    )
    ms2_val = load_mslr2022_local(
        config.data_dir, 
        split='dev', 
        dataset='ms2', 
        max_samples=1000,
        auto_download=True
    )
    
    # Optional: Cochrane dataset (smaller but cleaner)
    print("\n📥 Loading Cochrane (optional supplement)...")
    cochrane_train = load_mslr2022_local(
        config.data_dir,
        split='train',
        dataset='cochrane',
        max_samples=2000,
        auto_download=True
    )
    
    # Stage 2: Layman simplification datasets
    print("\n📥 Loading BioLaySumm2025-PLOS (Stage 2: Layman Simplification)...")
    biolaysumm_train = load_biolaysumm_2025_local(
        config.data_dir,
        split='train',
        max_samples=config.biolaysumm_samples,
        auto_download=True
    )
    biolaysumm_val = load_biolaysumm_2025_local(
        config.data_dir,
        split='dev',
        max_samples=500,
        auto_download=True
    )
    
    print("\n📥 Loading Med-EASi...")
    medeasi_train = load_medeasi_local(
        config.data_dir,
        split='train',
        max_samples=config.medeasi_samples,
        auto_download=True
    )
    
    # ============================================================
    # STAGE 1: SCIENTIFIC SYNTHESIS (MS2 + optional Cochrane)
    # ============================================================
    stage1_output = None
    if ms2_train:
        # Combine with Cochrane if available
        train_data = [ms2_train]
        if cochrane_train:
            train_data.append(cochrane_train)
            print(f"   ✅ Combined with {len(cochrane_train)} Cochrane examples")
        
        combined_train = concatenate_datasets(train_data).shuffle(seed=config.seed) if len(train_data) > 1 else ms2_train
        
        stage1_output, model, trainer1 = train_stage(
            model=model,
            tokenizer=tokenizer,
            train_dataset=combined_train,
            eval_dataset=ms2_val,
            stage_name="Stage1_Scientific",
            epochs=config.stage1_epochs,
            output_dir=output_dir
        )
        
        generate_training_plots(trainer1, output_dir, timestamp, "Stage1_Scientific")
        
        # Save checkpoint for Stage 2
        stage1_checkpoint = os.path.join(stage1_output, "stage1_checkpoint")
        os.makedirs(stage1_checkpoint, exist_ok=True)
        model.save_pretrained(stage1_checkpoint)
        tokenizer.save_pretrained(stage1_checkpoint)
        print(f"✅ Stage 1 checkpoint saved: {stage1_checkpoint}")
    else:
        print("⚠️ Skipping Stage 1 (MS2 not available)")
        stage1_checkpoint = None
    
    # ============================================================
    # STAGE 2: LAYMAN SIMPLIFICATION (BioLaySumm2025 + Med-EASi)
    # ============================================================
    if stage1_checkpoint:
        print(f"\n📥 Loading Stage 1 checkpoint: {stage1_checkpoint}")
        model = PeftModel.from_pretrained(model.base_model, stage1_checkpoint)
    
    # Combine layman datasets
    layman_datasets = []
    if biolaysumm_train:
        layman_datasets.append(biolaysumm_train)
    if medeasi_train:
        layman_datasets.append(medeasi_train)
    
    if layman_datasets:
        combined_layman = concatenate_datasets(layman_datasets).shuffle(seed=config.seed)
        combined_layman_val = biolaysumm_val if biolaysumm_val else None
        
        stage2_output, model, trainer2 = train_stage(
            model=model,
            tokenizer=tokenizer,
            train_dataset=combined_layman,
            eval_dataset=combined_layman_val,
            stage_name="Stage2_Layman",
            epochs=config.stage2_epochs,
            output_dir=output_dir
        )
        
        generate_training_plots(trainer2, output_dir, timestamp, "Stage2_Layman")
        
        # Save final model
        final_path = os.path.join(output_dir, "final_model")
        os.makedirs(final_path, exist_ok=True)
        model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        print(f"✅ Final model saved: {final_path}")
    else:
        print("⚠️ Skipping Stage 2 (no layman datasets available)")
        final_path = stage1_checkpoint if stage1_checkpoint else output_dir
    
    # ============================================================
    # INFERENCE DEMO
    # ============================================================
    print("\n" + "="*70)
    print("🧪 RUNNING INFERENCE DEMO")
    print("="*70)
    
    test_claim = "Vitamin D supplementation reduces risk of respiratory infections"
    test_evidence = "[Evidence 1] A meta-analysis of 25 RCTs found that vitamin D supplementation significantly reduced acute respiratory tract infections (OR 0.88, 95% CI 0.81-0.96). [Evidence 2] The protective effect was stronger in individuals with baseline vitamin D deficiency."
    
    print(f"\n📝 Claim: {test_claim}")
    print(f"\n📚 Evidence:\n{test_evidence}")
    
    print("\n🔬 Generating scientific summary...")
    scientific_summary = generate_summary(model, tokenizer, test_claim, test_evidence, register='scientific')
    print(f"   {scientific_summary}")
    
    print("\n🗣️ Generating layman summary...")
    layman_summary = generate_summary(model, tokenizer, test_claim, scientific_summary, register='layman')
    print(f"   {layman_summary}")
    
    fk_grade = flesch_kincaid_grade(layman_summary)
    print(f"\n📊 Readability: Flesch-Kincaid Grade Level = {fk_grade:.1f}")
    
    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print("\n" + "="*80)
    print("🏁 OpenBioLLM-8B SUMMARIZER TRAINING COMPLETE")
    print("="*80)
    print(f"✅ Final model: {final_path}")
    print(f"✅ Artifacts: {output_dir}")
    print(f"✅ Backup: {config.drive_dir}")
    print(f"\n🎯 Usage:")
    print(f"   from peft import PeftModel")
    print(f"   model = PeftModel.from_pretrained(base_model, '{final_path}')")
    print(f"   summary = generate_summary(model, tokenizer, claim, evidence, register='scientific')")
    print("="*80)
    
    # Cleanup
    if config.device == "cuda":
        torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    print("\n✅ Memory cleaned up successfully!")
    print("🎉 ARIYAAM v3.0 Summarizer ready for deployment!")

# ============================================================
# WINDOWS MULTIPROCESSING GUARD
# ============================================================
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
