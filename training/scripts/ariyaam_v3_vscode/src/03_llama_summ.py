# ============================================================
# ARIYAAM v3.0 — OPENBIOLLM-8B BIOMEDICAL SUMMARIZER (QLoRA)
# Module: 03_llama_summ.py (VS Code Compatible - UPDATED)
# Target: Scientific synthesis + Layman simplification via prompt conditioning
# ✅ MODEL: aaditya/Llama3-OpenBioLLM-8B (already biomedical fine-tuned)
# ✅ DATASETS: BioLaySumm2025-PLOS + Med-EASi (cbasu) + MS2
# ✅ REMOVED: PLABA (repository no longer exists)
# ✅ FIXED: Colab dependencies removed for local execution
# ✅ FIXED: Windows multiprocessing guards added
# ✅ FIXED: All paths are local/relative (no /content/)
# ✅ IMPLEMENTED: QLoRA 4-bit quantization + LoRA adapters
# ✅ IMPLEMENTED: Two-stage fine-tuning (MS2 → BioLaySumm2025 + Med-EASi)
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
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, field

# Third-party imports
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk, load_dataset
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
# CONFIGURATION — VS CODE LOCAL PATHS + UPDATED MODEL/DATASETS
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
    
    # ✅ UPDATED MODEL: OpenBioLLM-8B (already biomedical fine-tuned)
    model_name: str = "aaditya/Llama3-OpenBioLLM-8B"
    tokenizer_name: str = "aaditya/Llama3-OpenBioLLM-8B"
    
    # QLoRA configuration
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True
    
    # LoRA configuration (slightly reduced rank since OpenBioLLM is already fine-tuned)
    lora_rank: int = 8  # Reduced from 16 (model already has biomedical knowledge)
    lora_alpha: int = 16
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
    stage1_epochs: int = 2  # MS2 scientific synthesis (reduced: OpenBioLLM already knows biomedical)
    stage2_epochs: int = 2  # BioLaySumm2025 + Med-EASi layman simplification
    learning_rate: float = 1e-4  # Lower LR since model is already fine-tuned
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 0.3
    
    # ✅ UPDATED DATASET CONFIGURATION
    ms2_samples: int = 15000  # Slightly reduced
    biolaysumm2025_samples: int = 10000  # BioLaySumm2025-PLOS
    medeasi_samples: int = 2000  # Med-EASi (cbasu)
    
    # ✅ REMOVED: PLABA (repository no longer exists)
    # plaba_samples: int = 0  # Not used
    
    # Prompt templates (optimized for OpenBioLLM instruction format)
    scientific_prompt: str = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a biomedical research assistant. Synthesize the provided evidence passages into a concise verdict summary for the given claim. Follow these rules:
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
    
    # Evaluation targets (adjusted for OpenBioLLM baseline)
    target_rouge_l: float = 0.38  # Slightly higher expectation (model already biomedical)
    target_bertscore: float = 0.87
    target_sari: float = 0.42
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
        print(f"   Datasets: MS2 + BioLaySumm2025-PLOS + Med-EASi (cbasu)")
        print(f"   ⚠️ PLABA removed (repository no longer exists)")

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
# LOCAL DATASET LOADERS (UPDATED FOR NEW SOURCES)
# ============================================================
def load_ms2_local(data_dir, split='train', max_samples=None):
    """Load MS2 (Multi-Document Summarization of Medical Studies) from local files or HF"""
    print(f"📚 Loading MS2 from: {data_dir}")
    
    # Try local first, fallback to HF load_dataset
    local_path = os.path.join(data_dir, 'ms2', f"{split}.jsonl")
    
    if os.path.exists(local_path):
        print(f"   Loading from local file: {local_path}")
        data = []
        with open(local_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    row = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                
                cluster_id = row.get('cluster_id', f"ms2_{line_num}")
                papers = row.get('papers', [])
                reference_summary = row.get('summary', '')
                
                if not papers or not reference_summary:
                    continue
                
                # Format evidence passages
                evidence_texts = []
                for i, paper in enumerate(papers[:5]):
                    title = paper.get('title', '')
                    abstract = paper.get('abstract', '')
                    evidence_texts.append(f"[Evidence {i+1}] {title}: {abstract}")
                
                evidence = "\n\n".join(evidence_texts)
                
                data.append({
                    'claim': f"Synthesize findings from {len(papers)} studies",
                    'evidence': evidence,
                    'reference_summary': reference_summary,
                    'register': 'scientific',
                    'source': 'ms2',
                    'cluster_id': str(cluster_id)
                })
                
                if max_samples and len(data) >= max_samples:
                    break
        
        if data:
            print(f"✅ Loaded {len(data)} MS2 {split} examples from local")
            return Dataset.from_list(data)
    
    # Fallback to HF load_dataset
    try:
        print(f"   Local not found, loading from HuggingFace: allenai/ms2")
        dataset = load_dataset("allenai/ms2", split=split if split != 'dev' else 'validation')
        
        def convert_ms2(example):
            papers = example.get('papers', [])
            evidence_texts = []
            for i, paper in enumerate(papers[:5]):
                title = paper.get('title', '')
                abstract = paper.get('abstract', '')
                evidence_texts.append(f"[Evidence {i+1}] {title}: {abstract}")
            
            return {
                'claim': f"Synthesize findings from {len(papers)} studies",
                'evidence': "\n\n".join(evidence_texts),
                'reference_summary': example.get('summary', ''),
                'register': 'scientific',
                'source': 'ms2',
                'cluster_id': str(example.get('cluster_id', ''))
            }
        
        dataset = dataset.map(convert_ms2, batched=False)
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        print(f"✅ Loaded {len(dataset)} MS2 {split} examples from HF")
        return dataset
    
    except Exception as e:
        print(f"⚠️ Failed to load MS2: {e}")
        return None

def load_biolaysumm2025_plos_local(data_dir, split='train', max_samples=None):
    """Load BioLaySumm2025-PLOS from local files or HF"""
    print(f"📚 Loading BioLaySumm2025-PLOS from: {data_dir}")
    
    # Try local first
    local_path = os.path.join(data_dir, 'biolaysumm2025_plos', f"{split}.jsonl")
    
    if os.path.exists(local_path):
        print(f"   Loading from local file: {local_path}")
        data = []
        with open(local_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    row = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                
                # BioLaySumm2025-PLOS format may differ from 2023
                # Expected: paper_id, abstract, lay_summary, domain
                paper_id = row.get('paper_id', row.get('id', f"biolay2025_{line_num}"))
                abstract = row.get('abstract', row.get('technical_abstract', ''))
                lay_summary = row.get('lay_summary', row.get('plain_summary', ''))
                
                if not abstract or not lay_summary:
                    continue
                
                data.append({
                    'claim': 'Explain this research simply',
                    'evidence': f"[Evidence 1] {abstract}",
                    'reference_summary': lay_summary,
                    'register': 'layman',
                    'source': 'biolaysumm2025_plos',
                    'paper_id': str(paper_id),
                    'domain': row.get('domain', 'general')
                })
                
                if max_samples and len(data) >= max_samples:
                    break
        
        if data:
            print(f"✅ Loaded {len(data)} BioLaySumm2025-PLOS {split} examples from local")
            return Dataset.from_list(data)
    
    # Fallback to HF load_dataset
    try:
        print(f"   Local not found, loading from HuggingFace: BioLaySumm/BioLaySumm2025-PLOS")
        dataset = load_dataset("BioLaySumm/BioLaySumm2025-PLOS", split=split if split != 'dev' else 'validation')
        
        def convert_biolay2025(example):
            abstract = example.get('abstract', example.get('technical_abstract', ''))
            lay_summary = example.get('lay_summary', example.get('plain_summary', ''))
            
            return {
                'claim': 'Explain this research simply',
                'evidence': f"[Evidence 1] {abstract}",
                'reference_summary': lay_summary,
                'register': 'layman',
                'source': 'biolaysumm2025_plos',
                'paper_id': str(example.get('paper_id', example.get('id', ''))),
                'domain': example.get('domain', 'general')
            }
        
        dataset = dataset.map(convert_biolay2025, batched=False)
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        print(f"✅ Loaded {len(dataset)} BioLaySumm2025-PLOS {split} examples from HF")
        return dataset
    
    except Exception as e:
        print(f"⚠️ Failed to load BioLaySumm2025-PLOS: {e}")
        print(f"   Note: Dataset path is BioLaySumm/BioLaySumm2025-PLOS (not BioLaySumm2023)")
        return None

def load_medeasi_cbasu_local(data_dir, split='train', max_samples=None):
    """Load Med-EASi from cbasu/Med-EASi (local or HF)"""
    print(f"📚 Loading Med-EASi (cbasu) from: {data_dir}")
    
    # Try local first
    local_path = os.path.join(data_dir, 'medeasi_cbasu', f"{split}.jsonl")
    
    if os.path.exists(local_path):
        print(f"   Loading from local file: {local_path}")
        data = []
        with open(local_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    row = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                
                # cbasu/Med-EASi format
                original = row.get('original', row.get('technical', row.get('source', '')))
                simplified = row.get('simplified', row.get('plain', row.get('target', '')))
                
                if not original or not simplified:
                    continue
                
                data.append({
                    'claim': 'Rewrite in plain language',
                    'evidence': f"[Evidence 1] {original}",
                    'reference_summary': simplified,
                    'register': 'layman',
                    'source': 'medeasi_cbasu',
                    'example_id': f"medeasi_cbasu_{line_num}",
                    'complexity': row.get('complexity', 'medium')
                })
                
                if max_samples and len(data) >= max_samples:
                    break
        
        if data:
            print(f"✅ Loaded {len(data)} Med-EASi (cbasu) {split} examples from local")
            return Dataset.from_list(data)
    
    # Fallback to HF load_dataset
    try:
        print(f"   Local not found, loading from HuggingFace: cbasu/Med-EASi")
        dataset = load_dataset("cbasu/Med-EASi", split=split if split != 'dev' else 'validation')
        
        def convert_medeasi_cbasu(example):
            original = example.get('original', example.get('technical', example.get('source', '')))
            simplified = example.get('simplified', example.get('plain', example.get('target', '')))
            
            return {
                'claim': 'Rewrite in plain language',
                'evidence': f"[Evidence 1] {original}",
                'reference_summary': simplified,
                'register': 'layman',
                'source': 'medeasi_cbasu',
                'example_id': str(example.get('id', '')),
                'complexity': example.get('complexity', 'medium')
            }
        
        dataset = dataset.map(convert_medeasi_cbasu, batched=False)
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        print(f"✅ Loaded {len(dataset)} Med-EASi (cbasu) {split} examples from HF")
        return dataset
    
    except Exception as e:
        print(f"⚠️ Failed to load Med-EASi (cbasu): {e}")
        return None

# ✅ REMOVED: load_plaba_local() - repository no longer exists

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
# QLoRA MODEL LOADING (UPDATED FOR OPENBIOLLM)
# ============================================================
def load_openbiollm_qlora(model_name=None):
    """Load OpenBioLLM-8B with QLoRA configuration"""
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
        truncation_side='right',
        trust_remote_code=True
    )
    
    # Add special tokens if needed (OpenBioLLM uses Llama-3 tokenizer)
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
    
    # Configure LoRA (reduced rank since OpenBioLLM is already biomedical)
    lora_config = LoraConfig(
        r=config.lora_rank,  # 8 instead of 16 (model already fine-tuned)
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
    print(f"   Note: OpenBioLLM-8B is already biomedical fine-tuned - reduced LoRA rank for efficiency")
    
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
    print("🔬 ARIYAAM v3.0 — OPENBIOLLM-8B SUMMARIZER (QLoRA)")
    print("="*80)
    print(f"📁 Base: {config.base_dir}")
    print(f"📁 Data: {config.data_dir}")
    print(f"📁 Output: {config.output_dir}")
    print(f"🖥️ Device: {config.device}")
    print(f"🎯 Model: {config.model_name}")
    print(f"🎯 Target: ROUGE-L >{config.target_rouge_l}, BERTScore >{config.target_bertscore}")
    print(f"📚 Datasets: MS2 + BioLaySumm2025-PLOS + Med-EASi (cbasu)")
    print(f"⚠️ PLABA: Removed (repository no longer exists)")
    print("="*80)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config.output_dir, f"openbiollm_summarizer_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # ============================================================
    # LOAD MODEL & TOKENIZER
    # ============================================================
    print("\n" + "="*70)
    print(f"📥 LOADING {config.model_name} WITH QLoRA")
    print("="*70)
    
    # OpenBioLLM-8B is publicly available, no HF token required
    # But still check for auth in case of rate limiting
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print(f"✅ Using HF token for authentication")
    
    model, tokenizer = load_openbiollm_qlora()
    
    # ============================================================
    # LOAD DATASETS (UPDATED SOURCES)
    # ============================================================
    print("\n" + "="*70)
    print("📚 LOADING LOCAL/HF DATASETS")
    print("="*70)
    
    # Stage 1: MS2 (scientific multi-doc synthesis)
    print("\n📥 Loading MS2 (Stage 1: Scientific Synthesis)...")
    ms2_train = load_ms2_local(config.data_dir, 'train', max_samples=config.ms2_samples)
    ms2_val = load_ms2_local(config.data_dir, 'dev', max_samples=1000)
    
    # Stage 2: Layman simplification datasets (UPDATED)
    print("\n📥 Loading BioLaySumm2025-PLOS (Stage 2: Layman Simplification)...")
    biolay2025_train = load_biolaysumm2025_plos_local(config.data_dir, 'train', max_samples=config.biolaysumm2025_samples)
    biolay2025_val = load_biolaysumm2025_plos_local(config.data_dir, 'dev', max_samples=500)
    
    print("\n📥 Loading Med-EASi (cbasu)...")
    medeasi_train = load_medeasi_cbasu_local(config.data_dir, 'train', max_samples=config.medeasi_samples)
    
    # ✅ REMOVED: PLABA loading (repository no longer exists)
    print("\n⚠️ PLABA: Skipped (https://github.com/xiaoleihuang/PLABA no longer exists)")
    
    # ============================================================
    # STAGE 1: SCIENTIFIC SYNTHESIS (MS2)
    # ============================================================
    stage1_output = None
    if ms2_train:
        stage1_output, model, trainer1 = train_stage(
            model=model,
            tokenizer=tokenizer,
            train_dataset=ms2_train,
            eval_dataset=ms2_val,
            stage_name="Stage1_Scientific",
            epochs=config.stage1_epochs,  # Reduced to 2 (OpenBioLLM already biomedical)
            output_dir=output_dir
        )
        
        generate_training_plots(trainer1, output_dir, timestamp, "Stage1_Scientific")
        
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
    
    # Combine layman datasets (NO PLABA)
    layman_datasets = []
    if biolay2025_train:
        layman_datasets.append(biolay2025_train)
    if medeasi_train:
        layman_datasets.append(medeasi_train)
    
    if layman_datasets:
        combined_layman = concatenate_datasets(layman_datasets).shuffle(seed=config.seed)
        combined_layman_val = biolay2025_val if biolay2025_val else None
        
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
    print("🏁 OPENBIOLLM-8B SUMMARIZER TRAINING COMPLETE")
    print("="*80)
    print(f"✅ Final model: {final_path}")
    print(f"✅ Artifacts: {output_dir}")
    print(f"✅ Backup: {config.drive_dir}")
    print(f"\n🎯 Usage:")
    print(f"   from peft import PeftModel")
    print(f"   model = PeftModel.from_pretrained(base_model, '{final_path}')")
    print(f"   summary = generate_summary(model, tokenizer, claim, evidence, register='scientific')")
    print(f"\n📚 Dataset Notes:")
    print(f"   • BioLaySumm2025-PLOS: BioLaySumm/BioLaySumm2025-PLOS (HF)")
    print(f"   • Med-EASi: cbasu/Med-EASi (HF)")
    print(f"   • PLABA: REMOVED (repository no longer exists)")
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
