#!/usr/bin/env python3
"""
Nano Hybrid GPT: Complete comparison framework for MoR, CoE, and their combinations
One-click runner for all experiments
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import time
import json
from pathlib import Path
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class Config:
    # Model architecture
    vocab_size: int = 50257
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    
    # MoR specific
    max_recursion_depth: int = 3
    
    # CoE specific  
    n_experts: int = 8
    expert_k: int = 2
    chain_length: int = 2
    
    # Training
    batch_size: int = 8
    learning_rate: float = 1e-3
    max_iters: int = 500
    eval_interval: int = 50
    warmup_iters: int = 50
    
    # System
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    compile: bool = False

# ==============================================================================
# Components
# ==============================================================================

class Router(nn.Module):
    """Dynamic router for expert/depth selection"""
    
    def __init__(self, n_embd, n_choices, router_type='linear'):
        super().__init__()
        self.n_choices = n_choices
        self.router_type = router_type
        
        if router_type == 'linear':
            self.router = nn.Linear(n_embd, n_choices)
        else:
            self.router = nn.Sequential(
                nn.Linear(n_embd, n_embd // 2),
                nn.ReLU(),
                nn.Linear(n_embd // 2, n_choices)
            )
        
    def forward(self, x, k=1, training=True):
        # x: (B, T, C)
        logits = self.router(x)  # (B, T, n_choices)
        
        if training:
            # Use Gumbel softmax for differentiable sampling
            gumbel_noise = -torch.empty_like(logits).exponential_().log()
            logits = (logits + gumbel_noise) / 0.5
        
        if k == 1:
            # Discrete choice
            weights = torch.softmax(logits, dim=-1)
            if training:
                # Straight-through estimator
                hard_weights = torch.zeros_like(weights)
                indices = torch.argmax(weights, dim=-1, keepdim=True)
                hard_weights.scatter_(-1, indices, 1.0)
                weights = hard_weights + weights - weights.detach()
            else:
                indices = torch.argmax(weights, dim=-1)
                hard_weights = torch.zeros_like(weights)
                hard_weights.scatter_(-1, indices.unsqueeze(-1), 1.0)
                weights = hard_weights
        else:
            # Top-k selection
            weights = torch.softmax(logits, dim=-1)
            top_k_weights, top_k_indices = torch.topk(weights, k, dim=-1)
            top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
            
            # Create sparse weight tensor
            new_weights = torch.zeros_like(weights)
            new_weights.scatter_(-1, top_k_indices, top_k_weights)
            weights = new_weights
            
        return weights

class Expert(nn.Module):
    """Single expert network (FFN)"""
    
    def __init__(self, n_embd, n_inner=None):
        super().__init__()
        if n_inner is None:
            n_inner = 4 * n_embd
        
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_inner),
            nn.GELU(),
            nn.Linear(n_inner, n_embd),
            nn.Dropout(0.1),
        )
    
    def forward(self, x):
        return self.net(x)

class MultiHeadAttention(nn.Module):
    """Standard multi-head attention"""
    
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))
        
    def forward(self, x):
        B, T, C = x.shape
        
        # Calculate qkv
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2) 
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        
        return y

# ==============================================================================
# Base Model Classes
# ==============================================================================

class BaseGPT(nn.Module):
    """Base GPT class with common functionality"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        
        # Output head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Stats tracking
        self.reset_stats()
        
    def reset_stats(self):
        self.stats = {
            'total_flops': 0,
            'expert_usage': {},
            'routing_entropy': [],
            'layer_stats': []
        }
    
    def get_embeddings(self, idx):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(pos)
        
        return tok_emb + pos_emb
    
    def compute_loss(self, logits, targets):
        return nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

class StandardGPT(BaseGPT):
    """Standard Transformer baseline"""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        x = self.get_embeddings(idx)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = self.compute_loss(logits, targets)
            
        return logits, loss

class TransformerBlock(nn.Module):
    """Standard transformer block"""
    
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config.n_embd, config.n_head, config.block_size, config.dropout)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffn = Expert(config.n_embd)
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class MoRGPT(BaseGPT):
    """Mixture of Recursions model"""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Shared recursive block
        self.recursive_block = RecursiveBlock(config)
        
        # Depth router
        self.depth_router = Router(config.n_embd, config.max_recursion_depth, 'mlp')
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        x = self.get_embeddings(idx)
        B, T, C = x.shape
        
        # Get depth routing weights for each token
        depth_weights = self.depth_router(x, k=1, training=self.training)  # (B, T, max_depth)
        
        # Process tokens with different depths
        output = torch.zeros_like(x)
        
        for depth in range(self.config.max_recursion_depth):
            # Mask for tokens using this depth
            mask = depth_weights[:, :, depth].unsqueeze(-1)  # (B, T, 1)
            
            if mask.sum() > 0:
                # Process with this many recursive steps
                x_depth = x.clone()
                for step in range(depth + 1):
                    x_depth = self.recursive_block(x_depth)
                
                output += mask * x_depth
        
        x = self.ln_f(output)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = self.compute_loss(logits, targets)
            
        return logits, loss

class RecursiveBlock(nn.Module):
    """Shared recursive transformer block"""
    
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config.n_embd, config.n_head, config.block_size, config.dropout)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffn = Expert(config.n_embd)
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class CoEGPT(BaseGPT):
    """Chain of Experts model"""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Standard transformer blocks with CoE layers
        self.blocks = nn.ModuleList([
            CoEBlock(config) for _ in range(config.n_layer)
        ])
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        x = self.get_embeddings(idx)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = self.compute_loss(logits, targets)
            
        return logits, loss

class CoEBlock(nn.Module):
    """Chain of Experts transformer block"""
    
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config.n_embd, config.n_head, config.block_size, config.dropout)
        self.ln2 = nn.LayerNorm(config.n_embd)
        
        # Chain of Experts layer
        self.coe_layer = ChainOfExpertsLayer(config)
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.coe_layer(self.ln2(x))
        return x

class ChainOfExpertsLayer(nn.Module):
    """Chain of Experts layer implementation"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_experts = config.n_experts
        self.expert_k = config.expert_k
        self.chain_length = config.chain_length
        
        # Experts
        self.experts = nn.ModuleList([
            Expert(config.n_embd) for _ in range(self.n_experts)
        ])
        
        # Routers for each chain step
        self.routers = nn.ModuleList([
            Router(config.n_embd, self.n_experts, 'linear') 
            for _ in range(self.chain_length)
        ])
        
    def forward(self, x):
        # x: (B, T, C)
        residual_input = x
        
        for step in range(self.chain_length):
            # Get routing weights
            routing_weights = self.routers[step](x, k=self.expert_k, training=self.training)
            
            # Apply experts
            expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-2)  # (B, T, n_experts, C)
            
            # Weighted combination
            output = torch.sum(routing_weights.unsqueeze(-1) * expert_outputs, dim=-2)  # (B, T, C)
            
            # Residual connection
            x = x + output
            
        return x - residual_input  # Return the change

# ==============================================================================
# Hybrid Models
# ==============================================================================

class HybridNestedGPT(BaseGPT):
    """Hybrid Nested: MoR(CoE_experts)"""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Shared recursive CoE block
        self.recursive_coe_block = RecursiveCoEBlock(config)
        
        # Depth router
        self.depth_router = Router(config.n_embd, config.max_recursion_depth, 'mlp')
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        x = self.get_embeddings(idx)
        B, T, C = x.shape
        
        # Get depth routing weights
        depth_weights = self.depth_router(x, k=1, training=self.training)
        
        # Process with different recursive depths
        output = torch.zeros_like(x)
        
        for depth in range(self.config.max_recursion_depth):
            mask = depth_weights[:, :, depth].unsqueeze(-1)
            
            if mask.sum() > 0:
                x_depth = x.clone()
                for step in range(depth + 1):
                    x_depth = self.recursive_coe_block(x_depth)
                
                output += mask * x_depth
        
        x = self.ln_f(output)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = self.compute_loss(logits, targets)
            
        return logits, loss

class RecursiveCoEBlock(nn.Module):
    """Recursive block with CoE inside"""
    
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config.n_embd, config.n_head, config.block_size, config.dropout)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.coe_layer = ChainOfExpertsLayer(config)
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.coe_layer(self.ln2(x))
        return x

class HybridSerialGPT(BaseGPT):
    """Hybrid Serial: MoR_layer + CoE_layer alternating"""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Alternating MoR and CoE blocks
        self.blocks = nn.ModuleList()
        for i in range(config.n_layer):
            if i % 2 == 0:
                self.blocks.append(MoRBlock(config))
            else:
                self.blocks.append(CoEBlock(config))
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        x = self.get_embeddings(idx)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = self.compute_loss(logits, targets)
            
        return logits, loss

class MoRBlock(nn.Module):
    """Single MoR block for serial hybrid"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config.n_embd, config.n_head, config.block_size, config.dropout)
        self.ln2 = nn.LayerNorm(config.n_embd)
        
        # Shared FFN
        self.shared_ffn = Expert(config.n_embd)
        
        # Depth router
        self.depth_router = Router(config.n_embd, config.max_recursion_depth, 'linear')
        
    def forward(self, x):
        # Attention
        x = x + self.attn(self.ln1(x))
        
        # MoR processing
        residual = x
        x_norm = self.ln2(x)
        
        # Route depths
        depth_weights = self.depth_router(x_norm, k=1, training=self.training)
        
        output = torch.zeros_like(x_norm)
        for depth in range(self.config.max_recursion_depth):
            mask = depth_weights[:, :, depth].unsqueeze(-1)
            if mask.sum() > 0:
                x_depth = x_norm.clone()
                for step in range(depth + 1):
                    x_depth = self.shared_ffn(x_depth)
                output += mask * x_depth
                
        return residual + output

class HybridUnifiedGPT(BaseGPT):
    """Hybrid Unified: Single router controlling both depth and experts"""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Unified blocks
        self.blocks = nn.ModuleList([
            UnifiedBlock(config) for _ in range(config.n_layer)
        ])
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        x = self.get_embeddings(idx)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = self.compute_loss(logits, targets)
            
        return logits, loss

class UnifiedBlock(nn.Module):
    """Unified block with single router for depth + experts"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config.n_embd, config.n_head, config.block_size, config.dropout)
        self.ln2 = nn.LayerNorm(config.n_embd)
        
        # Experts
        self.experts = nn.ModuleList([
            Expert(config.n_embd) for _ in range(config.n_experts)
        ])
        
        # Unified router: outputs depth + expert selection
        total_choices = config.max_recursion_depth * config.n_experts
        self.unified_router = Router(config.n_embd, total_choices, 'mlp')
        
    def forward(self, x):
        # Attention
        x = x + self.attn(self.ln1(x))
        
        # Unified routing
        residual = x
        x_norm = self.ln2(x)
        
        # Get unified routing weights
        unified_weights = self.unified_router(x_norm, k=1, training=self.training)  # (B, T, depth*experts)
        
        # Reshape to separate depth and expert dimensions
        B, T, _ = unified_weights.shape
        routing_matrix = unified_weights.view(B, T, self.config.max_recursion_depth, self.config.n_experts)
        
        output = torch.zeros_like(x_norm)
        
        for depth in range(self.config.max_recursion_depth):
            for expert_idx in range(self.config.n_experts):
                weight = routing_matrix[:, :, depth, expert_idx].unsqueeze(-1)  # (B, T, 1)
                
                if weight.sum() > 0:
                    # Apply expert with this depth
                    x_processed = x_norm.clone()
                    for step in range(depth + 1):
                        x_processed = self.experts[expert_idx](x_processed)
                    
                    output += weight * x_processed
        
        return residual + output

# ==============================================================================
# Data and Training
# ==============================================================================

class TinyDataset(Dataset):
    """Simple synthetic dataset for quick testing"""
    
    def __init__(self, config, split='train'):
        self.config = config
        self.split = split
        
        # Generate synthetic sequences that require reasoning
        np.random.seed(42 if split == 'train' else 1337)
        
        # Create patterns that benefit from recursive/expert processing
        vocab_size = min(config.vocab_size, 1000)  # Use smaller vocab for speed
        seq_len = config.block_size
        
        if split == 'train':
            self.data = self._generate_reasoning_data(vocab_size, seq_len, 1000)
        else:
            self.data = self._generate_reasoning_data(vocab_size, seq_len, 200)
    
    def _generate_reasoning_data(self, vocab_size, seq_len, n_samples):
        """Generate sequences that require multi-step reasoning"""
        data = []
        
        for _ in range(n_samples):
            # Pattern 1: Simple arithmetic sequences
            if np.random.rand() < 0.3:
                seq = self._arithmetic_sequence(vocab_size, seq_len)
            # Pattern 2: Copy patterns with noise
            elif np.random.rand() < 0.6:
                seq = self._copy_pattern(vocab_size, seq_len)
            # Pattern 3: Random sequence
            else:
                seq = np.random.randint(1, vocab_size, seq_len)
            
            data.append(torch.tensor(seq, dtype=torch.long))
        
        return data
    
    def _arithmetic_sequence(self, vocab_size, seq_len):
        """Generate arithmetic progression"""
        start = np.random.randint(1, 10)
        step = np.random.randint(1, 5)
        seq = [(start + i * step) % vocab_size for i in range(seq_len)]
        return seq
    
    def _copy_pattern(self, vocab_size, seq_len):
        """Generate copy pattern with noise"""
        pattern_len = seq_len // 4
        pattern = np.random.randint(1, vocab_size, pattern_len)
        
        seq = []
        for i in range(seq_len):
            if i < pattern_len:
                seq.append(pattern[i])
            elif i < 2 * pattern_len:
                # Add noise
                seq.append(np.random.randint(1, vocab_size))
            else:
                # Repeat pattern
                seq.append(pattern[i % pattern_len])
        
        return seq
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq = self.data[idx]
        return seq[:-1], seq[1:]  # input, target

class Trainer:
    """Training and evaluation manager"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        # Setup datasets
        train_dataset = TinyDataset(config, 'train')
        val_dataset = TinyDataset(config, 'val')
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size, 
            shuffle=True,
            num_workers=0
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0
        )
        
    def train_model(self, model, model_name):
        """Train a single model"""
        print(f"\n{'='*50}")
        print(f"Training {model_name}")
        print(f"{'='*50}")
        
        model = model.to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        
        # Learning rate schedule
        def get_lr(iter_num):
            if iter_num < self.config.warmup_iters:
                return self.config.learning_rate * iter_num / self.config.warmup_iters
            else:
                return self.config.learning_rate * 0.1 ** ((iter_num - self.config.warmup_iters) / (self.config.max_iters - self.config.warmup_iters))
        
        model.train()
        train_losses = []
        val_losses = []
        times = []
        flops_per_iter = []
        
        start_time = time.time()
        
        for iter_num in range(self.config.max_iters):
            # Update learning rate
            lr = get_lr(iter_num)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # Training step
            model.train()
            for batch_idx, (x, y) in enumerate(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass with timing
                iter_start = time.time()
                logits, loss = model(x, y)
                iter_time = time.time() - iter_start
                
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
                times.append(iter_time)
                
                # Estimate FLOPs (rough approximation)
                flops = self._estimate_flops(model, x)
                flops_per_iter.append(flops)
                
                break  # One batch per iteration for simplicity
            
            # Validation
            if (iter_num + 1) % self.config.eval_interval == 0:
                model.eval()
                val_loss = 0
                val_count = 0
                
                with torch.no_grad():
                    for x, y in self.val_loader:
                        x, y = x.to(self.device), y.to(self.device)
                        logits, loss = model(x, y)
                        val_loss += loss.item()
                        val_count += 1
                        
                        if val_count >= 10:  # Quick validation
                            break
                
                avg_val_loss = val_loss / val_count
                val_losses.append(avg_val_loss)
                
                print(f"Iter {iter_num+1:4d} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {lr:.6f}")
        
        total_time = time.time() - start_time
        
        return {
            'model': model,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'times': times,
            'flops_per_iter': flops_per_iter,
            'total_time': total_time,
            'final_val_loss': val_losses[-1] if val_losses else float('inf'),
            'params': sum(p.numel() for p in model.parameters()),
            'model_name': model_name
        }
    
    def _estimate_flops(self, model, x):
        """Rough FLOP estimation"""
        # This is a very rough estimate - in practice you'd want more detailed profiling
        batch_size, seq_len = x.shape
        n_params = sum(p.numel() for p in model.parameters())
        
        # Rough approximation: 2 * params * batch_size * seq_len
        return 2 * n_params * batch_size * seq_len

def run_comparison(config):
    """Run complete comparison of all models"""
    
    print(f"Running comparison on {config.device}")
    print(f"Config: {config}")
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Define models to compare
    models_to_test = [
        ('StandardGPT', StandardGPT),
        ('MoR-GPT', MoRGPT),
        ('CoE-GPT', CoEGPT),
        ('Hybrid-Nested', HybridNestedGPT),
        ('Hybrid-Serial', HybridSerialGPT),
        ('Hybrid-Unified', HybridUnifiedGPT),
    ]
    
    results = {}
    
    # Train all models
    for model_name, model_class in models_to_test:
        try:
            model = model_class(config)
            result = trainer.train_model(model, model_name)
            results[model_name] = result
            
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            continue
    
    # Analyze and visualize results
    analyze_results(results, config)
    
    return results

def analyze_results(results, config):
    """Analyze and visualize comparison results"""
    
    print(f"\n{'='*70}")
    print("COMPARISON RESULTS")
    print(f"{'='*70}")
    
    # Summary table
    print(f"{'Model':<15} {'Params':<10} {'Final Loss':<12} {'Time (s)':<10} {'Efficiency':<10}")
    print(f"{'-'*70}")
    
    for name, result in results.items():
        params = result['params'] / 1e6  # Convert to millions
        final_loss = result['final_val_loss']
        time_taken = result['total_time']
        efficiency = params / max(1e-6, final_loss)  # Params per unit loss
        
        print(f"{name:<15} {params:<10.2f}M {final_loss:<12.4f} {time_taken:<10.1f} {efficiency:<10.1f}")
    
    # Create visualizations
    create_visualizations(results, config)

def create_visualizations(results, config):
    """Create comparison plots"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Training curves
    for name, result in results.items():
        if result['val_losses']:
            x = np.arange(len(result['val_losses'])) * config.eval_interval
            ax1.plot(x, result['val_losses'], label=name, marker='o', markersize=3)
    
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Validation Loss')
    ax1.set_title('Training Curves Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Parameter efficiency
    names = list(results.keys())
    params = [results[name]['params'] / 1e6 for name in names]
    final_losses = [results[name]['final_val_loss'] for name in names]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    
    for i, name in enumerate(names):
        ax2.scatter(params[i], final_losses[i], 
                   s=100, c=[colors[i]], label=name, alpha=0.7)
    
    ax2.set_xlabel('Parameters (M)')
    ax2.set_ylabel('Final Validation Loss')
    ax2.set_title('Parameter Efficiency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Training time comparison
    times = [results[name]['total_time'] for name in names]
    bars = ax3.bar(range(len(names)), times, color=colors)
    ax3.set_xlabel('Models')
    ax3.set_ylabel('Training Time (s)')
    ax3.set_title('Training Time Comparison')
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels(names, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Efficiency frontier
    efficiency_x = []
    efficiency_y = []
    
    for name in names:
        # X-axis: computational cost (params * time)
        comp_cost = (results[name]['params'] / 1e6) * results[name]['total_time']
        # Y-axis: performance (1 / loss)
        performance = 1.0 / max(results[name]['final_val_loss'], 1e-6)
        
        efficiency_x.append(comp_cost)
        efficiency_y.append(performance)
    
    for i, name in enumerate(names):
        ax4.scatter(efficiency_x[i], efficiency_y[i], 
                   s=100, c=[colors[i]], label=name, alpha=0.7)
    
    ax4.set_xlabel('Computational Cost (M-params × seconds)')
    ax4.set_ylabel('Performance (1/loss)')
    ax4.set_title('Efficiency Frontier')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparison_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nResults saved to 'comparison_results.png'")

# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Nano Hybrid GPT Comparison')
    parser.add_argument('--max_iters', type=int, default=500, help='Training iterations')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--n_layer', type=int, default=4, help='Number of layers')
    parser.add_argument('--n_embd', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    
    args = parser.parse_args()
    
    # Create config
    config = Config()
    if args.device == 'auto':
        config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        config.device = args.device
    
    config.max_iters = args.max_iters
    config.batch_size = args.batch_size
    config.n_layer = args.n_layer
    config.n_embd = args.n_embd
    
    print("🚀 Starting Nano Hybrid GPT Comparison")
    print(f"📊 Will compare 6 different architectures:")
    print("   • StandardGPT (baseline)")
    print("   • MoR-GPT (Mixture of Recursions)")
    print("   • CoE-GPT (Chain of Experts)")
    print("   • Hybrid-Nested (MoR + CoE nested)")
    print("   • Hybrid-Serial (MoR + CoE alternating)")
    print("   • Hybrid-Unified (unified routing)")
    
    # Run comparison
    results = run_comparison(config)
    
    print("\n🎉 Comparison completed!")
    print("📈 Check 'comparison_results.png' for visualizations")
    
    return results

if __name__ == "__main__":
    results = main()
