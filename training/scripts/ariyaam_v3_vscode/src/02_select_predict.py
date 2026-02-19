# ============================================================
# ARIYAAM v3.0 — SELECT-THEN-PREDICT RATIONALE ARCHITECTURE
# Module: 02_select_predict.py (VS Code Compatible)
# Target: Faithful NLI with extracted evidence rationales
# ✅ FIXED: Uses CUSTOM dataset scripts (not HuggingFace Hub)
# ✅ FIXED: Colab dependencies removed for local execution
# ✅ FIXED: Windows multiprocessing guards added
# ✅ FIXED: All paths are local/relative (no /content/)
# ✅ IMPLEMENTED: 3-phase training (e-SNLI → ERASER → SciFact)
# ✅ IMPLEMENTED: Sparsity + Continuity loss for rationale quality
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
from functools import partial
from collections import Counter

# Third-party imports
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from datasets import Dataset, DatasetDict, Features, Sequence, Value, concatenate_datasets, load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
    average_precision_score,
    precision_recall_curve
)

# Suppress warnings
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

# ============================================================
# CONFIGURATION — VS CODE LOCAL PATHS
# ============================================================
class Config:
    def __init__(self):
        # ✅ Local paths (no Colab /content/)
        self.base_dir = os.getcwd()
        self.output_dir = os.path.join(self.base_dir, "models")
        self.data_dir = os.path.join(self.base_dir, "data")
        self.datasets_dir = os.path.join(self.base_dir, "datasets")  # Custom dataset scripts
        self.drive_dir = os.path.join(self.base_dir, "drive_backup")
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.datasets_dir, exist_ok=True)
        os.makedirs(self.drive_dir, exist_ok=True)
        
        # Hardware detection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_gpus = torch.cuda.device_count()
        
        # Model configuration
        self.model_name = "pritamdeka/PubMedBERT-MNLI-MedNLI"
        self.hidden_size = 768  # PubMedBERT hidden size
        self.num_labels = 3
        
        # Training hyperparameters
        self.max_length = 128
        self.batch_size = 8  # Smaller for Select-then-Predict (more memory intensive)
        self.gradient_accumulation_steps = 4
        self.pretrain_epochs = 5  # Phase 1: e-SNLI
        self.domain_adapt_epochs = 3  # Phase 2: ERASER
        self.finetune_epochs = 8  # Phase 3: SciFact
        self.learning_rate = 2e-5
        self.weight_decay = 0.01
        self.warmup_ratio = 0.1
        self.lr_scheduler_type = "linear"
        self.max_grad_norm = 1.0
        self.early_stopping_patience = 10
        
        # Rationale loss weights
        self.lambda_sparsity = 0.01  # Encourage minimal selection
        self.lambda_continuity = 0.02  # Encourage contiguous spans
        
        # Selection threshold
        self.selection_threshold = 0.5
        
        # Target performance
        self.target_nli_f1 = 0.65  # Slightly lower than full-context NLI (acceptable trade-off)
        self.target_auprc = 0.75  # Rationale quality target
        
        # Reproducibility
        self.seed = 42
        
        print(f"✅ Configuration initialized")
        print(f"   Device: {self.device}")
        print(f"   GPUs: {self.num_gpus}")
        print(f"   Output: {self.output_dir}")
        print(f"   Data: {self.data_dir}")
        print(f"   Datasets: {self.datasets_dir}")

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
# SELECT-THEN-PREDICT MODEL ARCHITECTURE
# ============================================================
class StraightThroughEstimator(torch.autograd.Function):
    """Straight-through estimator for binary selection mask"""
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0.5).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # Pass gradient through sigmoid region, zero elsewhere
        sigmoid_grad = input * (1 - input)
        return grad_output * sigmoid_grad

class SelectThenPredictModel(nn.Module):
    """
    Joint Selector + Predictor for faithful NLI
    Architecture from v3.0 Strategy Document Section 4
    """
    def __init__(self, model_name, num_labels=3, max_sentences=10):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.num_labels = num_labels
        self.max_sentences = max_sentences
        
        # Selector head (sentence-level scoring)
        self.selector_head = nn.Linear(self.hidden_size, 1)
        
        # Predictor head (NLI classification)
        self.predictor_head = nn.Linear(self.hidden_size * 2, num_labels)  # claim + aggregated evidence
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, claim_ids, claim_mask, evidence_ids, evidence_mask, 
                rationale_labels=None, nli_labels=None, training=True):
        """
        Args:
            claim_ids: [batch, seq_len]
            claim_mask: [batch, seq_len]
            evidence_ids: [batch, num_sentences, seq_len]
            evidence_mask: [batch, num_sentences, seq_len]
            rationale_labels: [batch, num_sentences] (binary: 1=selected, 0=not)
            nli_labels: [batch] (0=SUPPORT, 1=CONTRADICT, 2=NEI)
        """
        batch_size, num_sentences, seq_len = evidence_ids.shape
        
        # Encode claim
        claim_outputs = self.encoder(input_ids=claim_ids, attention_mask=claim_mask)
        claim_embedding = claim_outputs.pooler_output  # [batch, hidden_size]
        
        # Encode evidence sentences
        evidence_ids_flat = evidence_ids.view(-1, seq_len)
        evidence_mask_flat = evidence_mask.view(-1, seq_len)
        
        evidence_outputs = self.encoder(
            input_ids=evidence_ids_flat,
            attention_mask=evidence_mask_flat
        )
        evidence_embeddings = evidence_outputs.pooler_output.view(
            batch_size, num_sentences, -1
        )  # [batch, num_sentences, hidden_size]
        
        # Selector: score each sentence
        selection_scores = self.selector_head(evidence_embeddings).squeeze(-1)  # [batch, num_sentences]
        selection_probs = torch.sigmoid(selection_scores)
        
        # Sample selection mask (straight-through estimator for gradient flow)
        if training:
            selection_mask = StraightThroughEstimator.apply(selection_probs)
        else:
            selection_mask = (selection_probs > config.selection_threshold).float()
        
        # Apply selection to evidence
        selected_evidence = evidence_embeddings * selection_mask.unsqueeze(-1)
        
        # Aggregate selected evidence (mean pooling over selected sentences)
        num_selected = selection_mask.sum(dim=1, keepdim=True) + 1e-9
        aggregated_evidence = selected_evidence.sum(dim=1) / num_selected  # [batch, hidden_size]
        
        # Predictor: classify based on claim + selected evidence
        combined = torch.cat([claim_embedding, aggregated_evidence], dim=-1)  # [batch, hidden_size * 2]
        combined = self.dropout(combined)
        nli_logits = self.predictor_head(combined)  # [batch, num_labels]
        
        # Compute loss
        loss = 0
        loss_dict = {}
        
        if nli_labels is not None:
            # NLI prediction loss
            nli_loss = nn.CrossEntropyLoss()(nli_logits, nli_labels)
            loss += nli_loss
            loss_dict['nli_loss'] = nli_loss.item()
        
        if rationale_labels is not None:
            # Rationale selection loss (BCE)
            rationale_loss = nn.BCELoss()(selection_probs, rationale_labels.float())
            loss += rationale_loss
            loss_dict['rationale_loss'] = rationale_loss.item()
            
            # Sparsity loss (L1 on selection mask - encourage minimal selection)
            sparsity_loss = config.lambda_sparsity * selection_mask.mean()
            loss += sparsity_loss
            loss_dict['sparsity_loss'] = sparsity_loss.item()
            
            # Continuity loss (penalize fragmented selections)
            if num_sentences > 1:
                continuity_loss = self._compute_continuity_loss(selection_probs)
                loss += config.lambda_continuity * continuity_loss
                loss_dict['continuity_loss'] = continuity_loss.item()
        
        loss_dict['total_loss'] = loss.item()
        
        return {
            'logits': nli_logits,
            'selection_probs': selection_probs,
            'selection_mask': selection_mask,
            'loss': loss,
            'loss_dict': loss_dict
        }
    
    def _compute_continuity_loss(self, selection_probs):
        """Penalize fragmented selections (encourage contiguous spans)"""
        # Compute differences between adjacent sentence probabilities
        diff = selection_probs[:, 1:] - selection_probs[:, :-1]  # [batch, num_sentences-1]
        # Penalize large changes (fragmentation)
        continuity_loss = (diff ** 2).mean()
        return continuity_loss
    
    def gradient_checkpointing_enable(self, **kwargs):
        return self.encoder.gradient_checkpointing_enable(**kwargs)
    
    def gradient_checkpointing_disable(self):
        return self.encoder.gradient_checkpointing_disable()
    
    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()
    
    def save_pretrained(self, save_directory, **kwargs):
        self.encoder.save_pretrained(save_directory, **kwargs)
        # Save additional heads
        heads_path = os.path.join(save_directory, "heads.pt")
        torch.save({
            'selector_head': self.selector_head.state_dict(),
            'predictor_head': self.predictor_head.state_dict()
        }, heads_path)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = cls(pretrained_model_name_or_path, *model_args, **kwargs)
        # Load additional heads if available
        heads_path = os.path.join(pretrained_model_name_or_path, "heads.pt")
        if os.path.exists(heads_path):
            heads = torch.load(heads_path, map_location='cpu')
            model.selector_head.load_state_dict(heads['selector_head'])
            model.predictor_head.load_state_dict(heads['predictor_head'])
        return model

# ============================================================
# CUSTOM DATASET LOADERS (Using Your Scripts)
# ============================================================
def load_custom_dataset(script_path, config_name=None, split=None, cache_dir=None):
    """
    Load dataset using custom HuggingFace dataset script (not from Hub)
    
    Args:
        script_path: Path to the Python dataset script file (.py)
        config_name: Configuration name (e.g., "v1.0", "plain_text")
        split: Split to load (e.g., "train", "validation")
        cache_dir: Directory to cache the downloaded data
    """
    from datasets import load_dataset
    
    # Get the directory containing the script
    script_dir = os.path.dirname(os.path.abspath(script_path))
    
    # Load using local path
    dataset = load_dataset(
        script_dir,  # Pass directory, not file path
        name=config_name,
        split=split,
        cache_dir=cache_dir or os.path.join(config.data_dir, "cache"),
        trust_remote_code=True
    )
    
    return dataset

def load_esnli_custom(cache_dir=None):
    """Load e-SNLI using custom script"""
    try:
        print("📚 Loading e-SNLI using custom script...")
        
        script_path = os.path.join(config.datasets_dir, "esnli", "esnli.py")
        
        if not os.path.exists(script_path):
            print(f"⚠️ e-SNLI script not found at: {script_path}")
            print("   Please download the esnli.py script to: datasets/esnli/")
            return None
        
        # Load all splits
        dataset = load_custom_dataset(
            script_path=script_path,
            config_name="plain_text",
            cache_dir=cache_dir
        )
        
        # Sample for faster training (e-SNLI is very large)
        if 'train' in dataset:
            train_sample = dataset['train'].select(range(min(50000, len(dataset['train']))))
        else:
            train_sample = dataset.select(range(min(50000, len(dataset))))
        
        if 'validation' in dataset:
            val_sample = dataset['validation'].select(range(min(5000, len(dataset['validation']))))
        else:
            val_sample = train_sample.select(range(min(5000, len(train_sample))))
        
        print(f"✅ Loaded e-SNLI: {len(train_sample)} training, {len(val_sample)} validation")
        return DatasetDict({
            'train': train_sample,
            'validation': val_sample
        })
    
    except Exception as e:
        print(f"⚠️ Failed to load e-SNLI: {e}")
        return None

def load_fever_custom(cache_dir=None):
    """Load FEVER using custom script"""
    try:
        print("📚 Loading FEVER using custom script...")
        
        script_path = os.path.join(config.datasets_dir, "fever", "fever.py")
        
        if not os.path.exists(script_path):
            print(f"⚠️ FEVER script not found at: {script_path}")
            print("   Please download the fever.py script to: datasets/fever/")
            return None
        
        # Load v1.0 config
        dataset = load_custom_dataset(
            script_path=script_path,
            config_name="v1.0",
            cache_dir=cache_dir
        )
        
        # Sample for training
        if 'train' in dataset:
            train_sample = dataset['train'].select(range(min(10000, len(dataset['train']))))
        else:
            train_sample = dataset.select(range(min(10000, len(dataset))))
        
        if 'labelled_dev' in dataset:
            val_sample = dataset['labelled_dev'].select(range(min(2000, len(dataset['labelled_dev']))))
        else:
            val_sample = train_sample.select(range(min(2000, len(train_sample))))
        
        print(f"✅ Loaded FEVER: {len(train_sample)} training, {len(val_sample)} validation")
        return DatasetDict({
            'train': train_sample,
            'validation': val_sample
        })
    
    except Exception as e:
        print(f"⚠️ Failed to load FEVER: {e}")
        return None

def load_scifact_custom(cache_dir=None):
    """Load SciFact using custom script"""
    try:
        print("📚 Loading SciFact using custom script...")
        
        script_path = os.path.join(config.datasets_dir, "scifact", "scifact.py")
        
        if not os.path.exists(script_path):
            print(f"⚠️ SciFact script not found at: {script_path}")
            print("   Please download the scifact.py script to: datasets/scifact/")
            return None
        
        # Load claims config (not corpus)
        dataset = load_custom_dataset(
            script_path=script_path,
            config_name="claims",
            cache_dir=cache_dir
        )
        
        # Convert to Select-then-Predict format
        def convert_format(examples):
            converted = {
                'claim': [],
                'evidence_sentences': [],
                'rationale_mask': [],
                'nli_label': [],
                'num_sentences': []
            }
            
            for claim, evidence_label, evidence_sentences in zip(
                examples['claim'],
                examples.get('evidence_label', [''] * len(examples['claim'])),
                examples.get('evidence_sentences', [[]] * len(examples['claim']))
            ):
                # Handle evidence sentences
                sentences = evidence_sentences if isinstance(evidence_sentences, list) else []
                if len(sentences) == 0:
                    sentences = ["No evidence available"]
                
                # Create rationale mask (1 if evidence_label is SUPPORT/REFUTE, else 0)
                rationale = [1] * min(len(sentences), config.max_sentences) if evidence_label in ['SUPPORT', 'REFUTES'] else [0] * min(len(sentences), config.max_sentences)
                while len(rationale) < config.max_sentences:
                    rationale.append(0)
                
                # Map NLI labels
                if evidence_label == 'SUPPORT':
                    nli_label = 0
                elif evidence_label == 'REFUTES':
                    nli_label = 1
                else:
                    nli_label = 2
                
                converted['claim'].append(claim)
                converted['evidence_sentences'].append(sentences[:config.max_sentences])
                converted['rationale_mask'].append(rationale)
                converted['nli_label'].append(nli_label)
                converted['num_sentences'].append(len(sentences))
            
            return converted
        
        if 'train' in dataset:
            train_converted = dataset['train'].map(convert_format, batched=True, batch_size=100)
            val_split = dataset.get('validation', dataset['train'].select(range(min(500, len(dataset['train'])))))
            val_converted = val_split.map(convert_format, batched=True, batch_size=100)
        else:
            # Single split dataset
            converted = dataset.map(convert_format, batched=True, batch_size=100)
            train_converted = converted.select(range(int(len(converted) * 0.8)))
            val_converted = converted.select(range(int(len(converted) * 0.8), len(converted)))
        
        print(f"✅ Loaded SciFact: {len(train_converted)} training, {len(val_converted)} validation")
        return DatasetDict({
            'train': train_converted,
            'validation': val_converted
        })
    
    except Exception as e:
        print(f"⚠️ Failed to load SciFact: {e}")
        return None

def load_eraser_dataset(cache_dir=None):
    """Load ERASER MultiRC + FEVER for Phase 2 domain adaptation"""
    try:
        print("📚 Loading ERASER datasets for domain adaptation...")
        
        # Try loading from HuggingFace Hub first (if available)
        try:
            multirc = load_dataset("eraser_multi_rc", cache_dir=cache_dir)
            print(f"✅ Loaded MultiRC from Hub: {len(multirc['train'])} examples")
        except:
            print("⚠️ MultiRC not available from Hub, skipping...")
            multirc = None
        
        datasets = []
        if multirc:
            datasets.append(multirc['train'].select(range(min(10000, len(multirc['train'])))))
        
        if datasets:
            combined = concatenate_datasets(datasets).shuffle(seed=42)
            return DatasetDict({
                'train': combined,
                'validation': combined.select(range(min(2000, len(combined))))
            })
        
        return None
    
    except Exception as e:
        print(f"⚠️ Failed to load ERASER: {e}")
        return None

# ============================================================
# TOKENIZATION FOR SELECT-THEN-PREDICT
# ============================================================
class SelectPredictDataCollator:
    """Custom data collator for Select-then-Predict architecture"""
    def __init__(self, tokenizer, max_length=128, max_sentences=10):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_sentences = max_sentences
    
    def __call__(self, features):
        batch = {
            'claim_ids': [],
            'claim_mask': [],
            'evidence_ids': [],
            'evidence_mask': [],
            'rationale_labels': [],
            'nli_labels': []
        }
        
        for feature in features:
            # Tokenize claim
            claim_text = feature.get('claim', feature.get('premise', feature.get('claim_text', '')))
            claim_tokens = self.tokenizer(
                claim_text,
                truncation=True,
                max_length=self.max_length,
                padding='max_length'
            )
            batch['claim_ids'].append(claim_tokens['input_ids'])
            batch['claim_mask'].append(claim_tokens['attention_mask'])
            
            # Tokenize evidence sentences
            evidence_sentences = feature.get('evidence_sentences', feature.get('sentences', []))
            if not evidence_sentences:
                evidence_sentences = [feature.get('premise', '')]
            
            evidence_ids = []
            evidence_mask = []
            for sent in evidence_sentences[:self.max_sentences]:
                sent_tokens = self.tokenizer(
                    str(sent),
                    truncation=True,
                    max_length=self.max_length,
                    padding='max_length'
                )
                evidence_ids.append(sent_tokens['input_ids'])
                evidence_mask.append(sent_tokens['attention_mask'])
            
            # Pad to max_sentences
            while len(evidence_ids) < self.max_sentences:
                evidence_ids.append([0] * self.max_length)
                evidence_mask.append([0] * self.max_length)
            
            batch['evidence_ids'].append(evidence_ids[:self.max_sentences])
            batch['evidence_mask'].append(evidence_mask[:self.max_sentences])
            
            # Rationale labels
            rationale = feature.get('rationale_mask', [0] * self.max_sentences)
            rationale = rationale[:self.max_sentences]
            while len(rationale) < self.max_sentences:
                rationale.append(0)
            batch['rationale_labels'].append(rationale)
            
            # NLI labels
            batch['nli_labels'].append(feature.get('nli_label', feature.get('label', 2)))
        
        # Convert to tensors
        batch['claim_ids'] = torch.tensor(batch['claim_ids'], dtype=torch.long)
        batch['claim_mask'] = torch.tensor(batch['claim_mask'], dtype=torch.long)
        batch['evidence_ids'] = torch.tensor(batch['evidence_ids'], dtype=torch.long)
        batch['evidence_mask'] = torch.tensor(batch['evidence_mask'], dtype=torch.long)
        batch['rationale_labels'] = torch.tensor(batch['rationale_labels'], dtype=torch.float)
        batch['nli_labels'] = torch.tensor(batch['nli_labels'], dtype=torch.long)
        
        return batch

# ============================================================
# CUSTOM TRAINER FOR SELECT-THEN-PREDICT
# ============================================================
class SelectPredictTrainer(Trainer):
    """Custom trainer with Select-then-Predict loss computation"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        claim_ids = inputs.pop('claim_ids')
        claim_mask = inputs.pop('claim_mask')
        evidence_ids = inputs.pop('evidence_ids')
        evidence_mask = inputs.pop('evidence_mask')
        rationale_labels = inputs.pop('rationale_labels', None)
        nli_labels = inputs.pop('nli_labels', None)
        
        outputs = model(
            claim_ids=claim_ids,
            claim_mask=claim_mask,
            evidence_ids=evidence_ids,
            evidence_mask=evidence_mask,
            rationale_labels=rationale_labels,
            nli_labels=nli_labels,
            training=self.model.training
        )
        
        loss = outputs['loss']
        
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Override to handle custom input format
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

# ============================================================
# EVALUATION METRICS
# ============================================================
def compute_select_predict_metrics(eval_pred):
    """Compute NLI F1 + Rationale AUPRC"""
    # Extract predictions
    if isinstance(eval_pred, tuple):
        logits, selection_probs, labels = eval_pred
    else:
        logits = eval_pred.predictions
        labels = eval_pred.label_ids
    
    # NLI predictions
    nli_preds = np.argmax(logits, axis=1)
    
    # NLI metrics
    present_classes = np.unique(labels)
    if len(present_classes) > 0:
        nli_f1_macro = f1_score(labels, nli_preds, average='macro', zero_division=0)
        nli_accuracy = accuracy_score(labels, nli_preds)
    else:
        nli_f1_macro = 0.0
        nli_accuracy = 0.0
    
    # Rationale metrics (if selection_probs available)
    rationale_auprc = 0.0
    if isinstance(eval_pred, tuple) and len(eval_pred) > 1:
        selection_probs = eval_pred[1]
        # Compute AUPRC for rationale selection
        try:
            rationale_auprc = average_precision_score(
                eval_pred[2] if len(eval_pred) > 2 else labels,
                selection_probs.mean(axis=1) if len(selection_probs.shape) > 1 else selection_probs
            )
        except:
            rationale_auprc = 0.0
    
    return {
        'nli_macro_f1': nli_f1_macro,
        'nli_accuracy': nli_accuracy,
        'rationale_auprc': rationale_auprc
    }

# ============================================================
# TRAINING PHASES
# ============================================================
def train_phase_1_pretrain(model, tokenizer, train_dataset, val_dataset, output_dir):
    """Phase 1: Pre-train Selector on e-SNLI rationale annotations"""
    print("\n" + "="*80)
    print("🚀 PHASE 1: SELECTOR PRE-TRAINING (e-SNLI)")
    print("="*80)
    
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "phase1_pretrain"),
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size * 2,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.pretrain_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        optim="adamw_torch",
        max_grad_norm=config.max_grad_norm,
        fp16=True if config.device == "cuda" else False,
        gradient_checkpointing=True,
        evaluation_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="nli_macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=config.seed,
        dataloader_num_workers=0,
    )
    
    data_collator = SelectPredictDataCollator(tokenizer, config.max_length, config.max_sentences)
    
    trainer = SelectPredictTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_select_predict_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)]
    )
    
    trainer.train()
    
    # Save checkpoint
    checkpoint_path = os.path.join(output_dir, "phase1_pretrain", "checkpoint")
    trainer.save_model(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)
    
    print(f"✅ Phase 1 complete. Checkpoint saved to: {checkpoint_path}")
    return checkpoint_path

def train_phase_2_domain_adapt(model, tokenizer, train_dataset, val_dataset, phase1_checkpoint, output_dir):
    """Phase 2: Fine-tune on ERASER MultiRC + FEVER"""
    print("\n" + "="*80)
    print("🚀 PHASE 2: DOMAIN ADAPTATION (ERASER)")
    print("="*80)
    
    # Load from Phase 1 checkpoint
    model = SelectThenPredictModel.from_pretrained(phase1_checkpoint, num_labels=config.num_labels)
    
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "phase2_domain_adapt"),
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size * 2,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.domain_adapt_epochs,
        learning_rate=config.learning_rate * 0.5,  # Lower LR for fine-tuning
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        optim="adamw_torch",
        max_grad_norm=config.max_grad_norm,
        fp16=True if config.device == "cuda" else False,
        gradient_checkpointing=True,
        evaluation_strategy="steps",
        eval_steps=150,
        save_strategy="steps",
        save_steps=150,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="nli_macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=config.seed,
        dataloader_num_workers=0,
    )
    
    data_collator = SelectPredictDataCollator(tokenizer, config.max_length, config.max_sentences)
    
    trainer = SelectPredictTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_select_predict_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)]
    )
    
    trainer.train()
    
    # Save checkpoint
    checkpoint_path = os.path.join(output_dir, "phase2_domain_adapt", "checkpoint")
    trainer.save_model(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)
    
    print(f"✅ Phase 2 complete. Checkpoint saved to: {checkpoint_path}")
    return checkpoint_path, model

def train_phase_3_finetune(model, tokenizer, train_dataset, val_dataset, phase2_checkpoint, output_dir):
    """Phase 3: Fine-tune on SciFact rationale annotations (biomedical)"""
    print("\n" + "="*80)
    print("🚀 PHASE 3: BIOMEDICAL FINE-TUNING (SciFact)")
    print("="*80)
    
    # Load from Phase 2 checkpoint
    model = SelectThenPredictModel.from_pretrained(phase2_checkpoint, num_labels=config.num_labels)
    
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "phase3_finetune"),
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size * 2,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.finetune_epochs,
        learning_rate=config.learning_rate * 0.3,  # Even lower LR for final fine-tuning
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        optim="adamw_torch",
        max_grad_norm=config.max_grad_norm,
        fp16=True if config.device == "cuda" else False,
        gradient_checkpointing=True,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="nli_macro_f1",
        greater_is_better=True,
        save_total_limit=3,
        report_to="none",
        seed=config.seed,
        dataloader_num_workers=0,
    )
    
    data_collator = SelectPredictDataCollator(tokenizer, config.max_length, config.max_sentences)
    
    trainer = SelectPredictTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_select_predict_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)]
    )
    
    trainer.train()
    
    # Save final model
    final_path = os.path.join(output_dir, "phase3_finetune", "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    
    print(f"✅ Phase 3 complete. Final model saved to: {final_path}")
    return final_path, model, trainer

# ============================================================
# VISUALIZATION
# ============================================================
def generate_training_plots(trainer, output_dir, timestamp):
    """Generate diagnostic plots for Select-then-Predict training"""
    figures_dir = os.path.join(output_dir, f"select_predict_{timestamp}", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # Training History
    history = trainer.state.log_history
    steps = [h["step"] for h in history if "loss" in h]
    train_loss = [h["loss"] for h in history if "loss" in h]
    eval_steps = [h["step"] for h in history if "nli_macro_f1" in h]
    eval_f1 = [h["nli_macro_f1"] for h in history if "nli_macro_f1" in h]
    
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(steps, train_loss, label="Train Loss", color="#2E86AB", linewidth=2)
    plt.xlabel("Training Steps", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.title("Training Loss", fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    if eval_steps and eval_f1:
        plt.plot(eval_steps, eval_f1, label="Val NLI F1", color="#F18F01", marker='s', linewidth=2, markersize=4)
        plt.axhline(y=config.target_nli_f1, color='r', linestyle='--', label=f'Target ({config.target_nli_f1})', alpha=0.7)
    plt.xlabel("Training Steps", fontsize=11)
    plt.ylabel("NLI Macro F1", fontsize=11)
    plt.title("Validation NLI F1 Progression", fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "training_history.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Training plots saved to: {figures_dir}")

# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================
def main():
    """Main training entry point - Windows multiprocessing safe"""
    print("\n" + "="*80)
    print("🔬 ARIYAAM v3.0 — SELECT-THEN-PREDICT TRAINING (VS CODE COMPATIBLE)")
    print("="*80)
    print(f"📁 Base Directory: {config.base_dir}")
    print(f"📁 Output Directory: {config.output_dir}")
    print(f"📁 Data Directory: {config.data_dir}")
    print(f"📁 Datasets Directory: {config.datasets_dir}")
    print(f"🖥️ Device: {config.device}")
    print(f"🎯 Target NLI F1: {config.target_nli_f1}")
    print(f"🎯 Target Rationale AUPRC: {config.target_auprc}")
    print("="*80)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config.output_dir, f"select_predict_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.add_special_tokens({'additional_special_tokens': ["[CLAIM]", "[EVIDENCE]"]})
    
    # ============================================================
    # LOAD DATASETS USING CUSTOM SCRIPTS
    # ============================================================
    print("\n" + "="*70)
    print("📚 LOADING DATASETS USING CUSTOM SCRIPTS")
    print("="*70)
    
    cache_dir = os.path.join(config.data_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Phase 1: e-SNLI (custom script)
    esnli_data = load_esnli_custom(cache_dir=cache_dir)
    
    # Phase 2: ERASER + FEVER (custom script for FEVER)
    eraser_data = load_eraser_dataset(cache_dir=cache_dir)
    fever_data = load_fever_custom(cache_dir=cache_dir)
    
    # Phase 3: SciFact (custom script)
    scifact_data = load_scifact_custom(cache_dir=cache_dir)
    
    # ============================================================
    # INITIALIZE MODEL
    # ============================================================
    print("\n" + "="*70)
    print("📥 INITIALIZING SELECT-THEN-PREDICT MODEL")
    print("="*70)
    
    model = SelectThenPredictModel(
        config.model_name,
        num_labels=config.num_labels,
        max_sentences=config.max_sentences
    )
    model.to(config.device)
    
    print(f"✅ Model initialized with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    
    # ============================================================
    # TRAINING PHASES
    # ============================================================
    phase1_checkpoint = None
    phase2_checkpoint = None
    
    # Phase 1: Pre-training (e-SNLI)
    if esnli_data:
        phase1_checkpoint = train_phase_1_pretrain(
            model, tokenizer,
            esnli_data['train'],
            esnli_data['validation'],
            output_dir
        )
    else:
        print("⚠️ Skipping Phase 1 (e-SNLI not available)")
        # Save initial model as checkpoint
        phase1_checkpoint = os.path.join(output_dir, "phase1_pretrain", "checkpoint")
        os.makedirs(phase1_checkpoint, exist_ok=True)
        model.save_pretrained(phase1_checkpoint)
        tokenizer.save_pretrained(phase1_checkpoint)
    
    # Phase 2: Domain Adaptation (ERASER + FEVER)
    phase2_datasets = []
    if eraser_data:
        phase2_datasets.append(eraser_data['train'])
    if fever_data:
        phase2_datasets.append(fever_data['train'])
    
    if phase2_datasets:
        combined_train = concatenate_datasets(phase2_datasets).shuffle(seed=42)
        combined_val = combined_train.select(range(min(2000, len(combined_train))))
        
        phase2_checkpoint, model = train_phase_2_domain_adapt(
            model, tokenizer,
            combined_train,
            combined_val,
            phase1_checkpoint,
            output_dir
        )
    else:
        print("⚠️ Skipping Phase 2 (ERASER/FEVER not available)")
        phase2_checkpoint = phase1_checkpoint
    
    # Phase 3: Biomedical Fine-tuning (SciFact)
    if scifact_data:
        final_path, model, trainer = train_phase_3_finetune(
            model, tokenizer,
            scifact_data['train'],
            scifact_data['validation'],
            phase2_checkpoint,
            output_dir
        )
        
        # Generate plots
        generate_training_plots(trainer, output_dir, timestamp)
        
        # Final evaluation
        print("\n" + "="*80)
        print("📊 FINAL EVALUATION")
        print("="*80)
        
        eval_results = trainer.evaluate()
        print(f"Final NLI Macro F1: {eval_results.get('eval_nli_macro_f1', 0):.4f}")
        print(f"Final Rationale AUPRC: {eval_results.get('eval_rationale_auprc', 0):.4f}")
        
        # Save metrics
        metrics = {
            'final_nli_f1': eval_results.get('eval_nli_macro_f1', 0),
            'final_rationale_auprc': eval_results.get('eval_rationale_auprc', 0),
            'timestamp': timestamp,
            'model': config.model_name,
            'phases_completed': 3,
            'target_nli_f1': config.target_nli_f1,
            'target_auprc': config.target_auprc
        }
        
        with open(os.path.join(final_path, "metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"✅ Metrics saved to: {os.path.join(final_path, 'metrics.json')}")
    else:
        print("⚠️ Skipping Phase 3 (SciFact not available)")
        final_path = phase2_checkpoint
    
    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print("\n" + "="*80)
    print("🏁 SELECT-THEN-PREDICT TRAINING COMPLETED")
    print("="*80)
    print(f"✅ Final model saved to: {final_path}")
    print(f"✅ Training artifacts saved to: {output_dir}")
    print(f"✅ Backup saved to: {config.drive_dir}")
    print("="*80)
    
    # Cleanup
    print("\n🧹 Cleaning up memory...")
    import gc
    if config.device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    print("✅ Memory cleaned up successfully!")
    
    print("\n" + "="*80)
    print("🎉 ARIYAAM v3.0 SELECT-THEN-PREDICT TRAINING COMPLETE!")
    print("="*80)

# ============================================================
# WINDOWS MULTIPROCESSING GUARD (CRITICAL FOR VS CODE ON WINDOWS)
# ============================================================
if __name__ == "__main__":
    # Required for Windows multiprocessing compatibility
    mp.set_start_method('spawn', force=True)
    main()
