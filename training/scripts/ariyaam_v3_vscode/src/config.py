# ============================================================
# ARIYAAM v3.0 — GLOBAL CONFIGURATION
# Module: config.py (VS Code Compatible)
# Purpose: Centralized configuration for all training scripts
# ✅ All paths match project tree structure
# ✅ All datasets verified available
# ✅ Hardware-aware optimizations for 10GB GPU
# ============================================================

import os
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path

@dataclass
class Paths:
    """All file paths — relative to project root"""
    # Base directories
    base_dir: str = field(default_factory=lambda: os.getcwd())
    output_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), "models"))
    data_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), "data"))
    drive_backup_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), "drive_backup"))
    
    # Dataset paths (matching your tree structure)
    scifact_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), "data", "scifact"))
    healthver_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), "data", "HealthVer", "data"))
    esnli_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), "data", "esnli"))
    fever_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), "data", "fever"))
    ms2_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), "data", "ms2"))
    biolaysumm_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), "data", "biolaysumm"))
    medeasi_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), "data", "medeasi"))
    plaba_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), "data", "plaba"))
    eraser_multirc_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), "data", "eraser_multirc"))
    
    # Specific dataset files
    scifact_train: str = field(default_factory=lambda: os.path.join(os.getcwd(), "data", "scifact", "claims_train.jsonl"))
    scifact_dev: str = field(default_factory=lambda: os.path.join(os.getcwd(), "data", "scifact", "claims_dev.jsonl"))
    scifact_test: str = field(default_factory=lambda: os.path.join(os.getcwd(), "data", "scifact", "claims_test.jsonl"))
    
    healthver_train: str = field(default_factory=lambda: os.path.join(os.getcwd(), "data", "HealthVer", "data", "healthver_train.csv"))
    healthver_dev: str = field(default_factory=lambda: os.path.join(os.getcwd(), "data", "HealthVer", "data", "healthver_dev.csv"))
    healthver_test: str = field(default_factory=lambda: os.path.join(os.getcwd(), "data", "HealthVer", "data", "healthver_test.csv"))
    
    esnli_train_1: str = field(default_factory=lambda: os.path.join(os.getcwd(), "data", "esnli", "esnli_train_1.csv"))
    esnli_train_2: str = field(default_factory=lambda: os.path.join(os.getcwd(), "data", "esnli", "esnli_train_2.csv"))
    esnli_dev: str = field(default_factory=lambda: os.path.join(os.getcwd(), "data", "esnli", "esnli_dev.csv"))
    esnli_test: str = field(default_factory=lambda: os.path.join(os.getcwd(), "data", "esnli", "esnli_test.csv"))
    
    fever_train: str = field(default_factory=lambda: os.path.join(os.getcwd(), "data", "fever", "train.jsonl"))
    fever_dev: str = field(default_factory=lambda: os.path.join(os.getcwd(), "data", "fever", "shared_task_dev.jsonl"))
    
    ms2_archive: str = field(default_factory=lambda: os.path.join(os.getcwd(), "data", "ms2", "mslr_data.tar.gz"))
    
    def __post_init__(self):
        """Create all directories if they don't exist"""
        for dir_path in [self.output_dir, self.data_dir, self.drive_backup_dir,
                        self.scifact_dir, self.healthver_dir, self.esnli_dir,
                        self.fever_dir, self.ms2_dir, self.biolaysumm_dir,
                        self.medeasi_dir, self.plaba_dir, self.eraser_multirc_dir]:
            os.makedirs(dir_path, exist_ok=True)

@dataclass
class Hardware:
    """Hardware-aware configuration"""
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    num_gpus: int = field(default_factory=lambda: torch.cuda.device_count())
    
    def __post_init__(self):
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            print(f"🔧 Detected GPU: {gpu_name} ({vram_gb:.1f}GB VRAM)")
            
            # Auto-configure based on VRAM
            if vram_gb < 12:
                print("⚠️ 10GB GPU detected — applying memory optimizations")
                self.is_10gb_gpu = True
            else:
                self.is_10gb_gpu = False
        else:
            print("⚠️ No GPU detected — using CPU (training will be slow)")
            self.is_10gb_gpu = True

@dataclass
class NLIConfig:
    """NLI Model Configuration (01_nli_v3.8.5.py)"""
    model_name: str = "pritamdeka/PubMedBERT-MNLI-MedNLI"
    max_length: int = 128
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    num_epochs: int = 15
    learning_rate: float = 3e-5
    target_macro_f1: float = 0.70
    target_per_class: int = 9000
    
    def apply_10gb_optimizations(self, hardware: Hardware):
        """Apply optimizations for 10GB GPU"""
        if hardware.is_10gb_gpu:
            self.batch_size = 8
            self.gradient_accumulation_steps = 8
            print("   ✅ NLI: batch_size=8, grad_accum=8 (10GB optimized)")

@dataclass
class SelectPredictConfig:
    """Select-then-Predict Configuration (02_select_predict.py)"""
    model_name: str = "pritamdeka/PubMedBERT-MNLI-MedNLI"
    max_length: int = 128
    max_sentences: int = 10
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    pretrain_epochs: int = 5  # e-SNLI
    domain_adapt_epochs: int = 3  # ERASER
    finetune_epochs: int = 8  # SciFact
    learning_rate: float = 2e-5
    lambda_sparsity: float = 0.01
    lambda_continuity: float = 0.02
    target_nli_f1: float = 0.65
    target_auprc: float = 0.75
    
    def apply_10gb_optimizations(self, hardware: Hardware):
        if hardware.is_10gb_gpu:
            self.batch_size = 4
            self.max_sentences = 5
            self.gradient_accumulation_steps = 8
            print("   ✅ Select-Predict: batch_size=4, max_sentences=5 (10GB optimized)")

@dataclass
class SummarizerConfig:
    """Llama-3 Summarizer Configuration (03_llama_summ.py)"""
    model_name: str = "aaditya/Llama3-OpenBioLLM-8B"
    tokenizer_name: str = "aaditya/Llama3-OpenBioLLM-8B"
    max_length: int = 2048
    max_new_tokens: int = 256
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    stage1_epochs: int = 2  # MS2
    stage2_epochs: int = 1  # BioLaySumm
    learning_rate: float = 3e-4
    ms2_samples: int = 10000
    biolaysumm_samples: int = 8000
    medeasi_samples: int = 300
    target_rouge_l: float = 0.35
    target_bertscore: float = 0.85
    
    def apply_10gb_optimizations(self, hardware: Hardware):
        if hardware.is_10gb_gpu:
            self.max_length = 2048
            self.lora_rank = 8
            self.gradient_accumulation_steps = 16
            print("   ✅ Summarizer: max_length=2048, lora_rank=8 (10GB optimized)")

@dataclass
class GEvalConfig:
    """G-Eval Judge Configuration (04_geval.py)"""
    model_name: str = "aaditya/Llama3-OpenBioLLM-8B"
    tokenizer_name: str = "aaditya/Llama3-OpenBioLLM-8B"
    max_length: int = 1024
    max_new_tokens: int = 128
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    lora_rank: int = 4
    lora_alpha: int = 8
    lora_dropout: float = 0.05
    epochs: int = 2
    learning_rate: float = 3e-4
    fib_samples: int = 200
    summeval_samples: int = 200
    medfactcheck_samples: int = 300
    synthetic_samples: int = 400
    target_spearman: float = 0.70
    baseline_agreement_threshold: float = 0.85
    
    def apply_10gb_optimizations(self, hardware: Hardware):
        if hardware.is_10gb_gpu:
            self.max_length = 1024
            self.lora_rank = 4
            self.batch_size = 1
            print("   ✅ G-Eval: max_length=1024, lora_rank=4 (10GB optimized)")

@dataclass
class DatasetConfig:
    """Dataset availability and sampling configuration"""
    # NLI Datasets
    scifact_available: bool = True
    healthver_available: bool = True
    pubmedqa_available: bool = True  # Auto-download from HF
    
    # Select-then-Predict Datasets
    esnli_available: bool = True
    fever_available: bool = True
    eraser_multirc_available: bool = False  # Optional
    
    # Summarizer Datasets
    ms2_available: bool = True  # Auto-download from HF or extract from tar.gz
    biolaysumm_available: bool = True  # Auto-download from HF
    medeasi_available: bool = True  # Auto-download from HF
    plaba_available: bool = False  # Repo no longer exists
    
    # G-Eval Datasets
    fib_available: bool = True  # Auto-download from HF
    summeval_available: bool = True  # Auto-download from HF
    medfactcheck_available: bool = False  # Manual download
    
    def verify_datasets(self, paths: Paths) -> Dict[str, bool]:
        """Check which datasets are available locally"""
        availability = {}
        
        # NLI
        availability['scifact'] = all(os.path.exists(getattr(paths, f'scifact_{split}')) 
                                     for split in ['train', 'dev', 'test'])
        availability['healthver'] = all(os.path.exists(getattr(paths, f'healthver_{split}')) 
                                       for split in ['train', 'dev', 'test'])
        
        # Select-then-Predict
        availability['esnli'] = all(os.path.exists(getattr(paths, f'esnli_{f}')) 
                                   for f in ['train_1', 'train_2', 'dev', 'test'])
        availability['fever'] = os.path.exists(paths.fever_train)
        
        # Summarizer
        availability['ms2'] = os.path.exists(paths.ms2_archive)
        
        print("\n📊 Dataset Availability:")
        for dataset, available in availability.items():
            status = "✅" if available else "⚠️"
            print(f"   {status} {dataset}")
        
        return availability

# ============================================================
# GLOBAL CONFIGURATION INSTANCE
# ============================================================
class Config:
    """Master configuration class"""
    def __init__(self):
        self.paths = Paths()
        self.hardware = Hardware()
        self.nli = NLIConfig()
        self.select_predict = SelectPredictConfig()
        self.summarizer = SummarizerConfig()
        self.geval = GEvalConfig()
        self.datasets = DatasetConfig()
        
        # Apply 10GB GPU optimizations automatically
        self.nli.apply_10gb_optimizations(self.hardware)
        self.select_predict.apply_10gb_optimizations(self.hardware)
        self.summarizer.apply_10gb_optimizations(self.hardware)
        self.geval.apply_10gb_optimizations(self.hardware)
        
        # Verify datasets
        self.dataset_availability = self.datasets.verify_datasets(self.paths)
        
        # HuggingFace token check
        self.hf_token = os.environ.get("HF_TOKEN")
        if not self.hf_token:
            print("\n⚠️ HF_TOKEN not set. Some datasets require authentication.")
            print("   Set with: export HF_TOKEN=your_token_here")
        
        print("\n✅ Configuration initialized successfully")

# Create global config instance
config = Config()

# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def get_device():
    """Get current device"""
    return config.hardware.device

def get_output_dir():
    """Get output directory"""
    return config.paths.output_dir

def get_data_dir():
    """Get data directory"""
    return config.paths.data_dir

def is_10gb_gpu():
    """Check if running on 10GB GPU"""
    return config.hardware.is_10gb_gpu

if __name__ == "__main__":
    print("="*80)
    print("ARIYAAM v3.0 — Configuration Test")
    print("="*80)
    print(f"Device: {config.hardware.device}")
    print(f"GPUs: {config.hardware.num_gpus}")
    print(f"Output Dir: {config.paths.output_dir}")
    print(f"Data Dir: {config.paths.data_dir}")
    print("="*80)
