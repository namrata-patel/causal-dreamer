# CausalDreamer: Temporal Image Generation through Causal Intervention Attention

**Anonymous code repository for ICME 2026 submission**

This repository contains the implementation of CausalDreamer, a framework for identity-preserving temporal image generation through causal intervention attention.

---

## 📋 Overview

CausalDreamer generates temporally ordered image sequences showing how causal events unfold while preserving subject identity. Given a cause description (e.g., "a dog running"), it produces 8 frames showing progressive temporal states (paws touching ground → body airborne → landing).

**Key Features:**
- Text-only training (no paired temporal visual data)
- Lightweight causal adapter (1.3M parameters)
- Causal intervention attention mechanism
- Identity-preserving temporal generation
- 4.7× better semantic diversity than ControlNet

---

## 🚀 Quick Start

### Prerequisites

```bash
Python >= 3.8
CUDA >= 11.3 (for GPU support)
```

### Installation

```bash
# Clone repository
git clone https://anonymous.4open.science/r/causal-dreamer-A501
cd causal-dreamer-A501

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Pretrained Models

```bash
# Download pretrained causal adapter
# Model Checkpoint is available in: models/causal_adapter_epoch_200.pt

# Stable Diffusion 1.5 will be automatically downloaded on first run
```

### Basic Usage

```bash
# Generate temporal sequence from a cause description
python inference.py \
  --cause "a dog running in a park" \
  --num_frames 8 \
  --output_dir outputs/

# Results will be saved in outputs/ directory
```

---

## 📁 Repository Structure

```
causal-dreamer/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── src/training/train_adapter.py            # Training script
├── src/inference/inference.py                # Inference script
├── src/inference/eval_metrics.py             # Evaluation code
├── src/models/
│   ├── causal_adapter.py       # Causal adapter architecture
│   ├── causal_intervention_attention.py
│   ├── causal_intervention_processor.py
│   └── causal_adapter_epoch_200.pt  # Pretrained weights (200 epochs)
├── data/
│   ├── CausalLite10K.csv      # Training dataset (10K cause-effect pairs)
│   └── test_counterfactual.csv     # 190 test scenarios
└── configs/
    └── config.yaml             # Configuration file
```

---

## 🎓 Training

### Prepare Dataset

The CausalLite10K dataset is provided in `data/CausalLite10K.json`:
- 10,000 cause-effect pairs
- 5,131 unique temporal effects
- Covers physical events, human actions, environmental phenomena

### Train Causal Adapter

```bash
python train_adapter.py \
  --config configs/config.yaml \
  --data_path data/CausalLite10K.json \
  --output_dir checkpoints/

# Training takes ~4 hours on a single NVIDIA A100 GPU
# Only the causal adapter (1.3M params) is trained
# CLIP encoder and Stable Diffusion remain frozen
```

### Training Configuration

Key hyperparameters (in `configs/config.yaml`):
```yaml
epochs: 200
batch_size: 32
learning_rate: 1e-4
optimizer: AdamW
clip_model: ViT-L/14
sd_version: stable-diffusion-v1-5
intervention_lambda: 0.7
```

---

## 🔬 Inference

### Generate Single Sequence

```bash
python inference.py \
  --cause "a glass tipping over" \
  --num_frames 8 \
  --checkpoint models/causal_adapter_epoch_200.pt \
  --output_dir outputs/glass_tipping/
```

### Generate Multiple Scenarios

```bash
# Use predefined test scenarios
python inference.py \
  --test_file data/test_scenarios.json \
  --checkpoint models/causal_adapter_epoch_200.pt \
  --output_dir outputs/test_results/
```

### Advanced Options

```bash
python inference.py \
  --cause "a cat jumping on a table" \
  --num_frames 8 \
  --top_k 8                    # Number of effects to retrieve
  --guidance_scale 7.5         # Classifier-free guidance
  --num_inference_steps 50     # DDIM sampling steps
  --seed 42                    # Random seed for reproducibility
  --output_dir outputs/cat_jumping/
```

---

## 📊 Evaluation

### Run Full Evaluation

Evaluate on 190 test scenarios with 6 metrics:

```bash
python eval_metrics.py \
  --results_dir outputs/test_results/ \
  --reference_dir data/test_scenarios/ \
  --output_file results/metrics.json
```

### Metrics Computed

- **DINO-TCS** (Identity Consistency): Subject preservation using DINOv2
- **SDS** (Semantic Diversity Score): Temporal variation between frames
- **CLIP** (Text-Image Alignment): Frame-effect correspondence
- **CCS** (Causal Consistency Score): Progression from cause to effect
- **CPS** (Causal Progression Score): Per-frame causal alignment
- **ETC** (Event Transition Coherence): Temporal transition coherence

### Baseline Comparisons

We compare against:
- Stable Diffusion v1.5
- SDXL
- ControlNet
- InstructPix2Pix
- AnimateDiff

---

## 🎯 Results

### Quantitative Results (on 190 test scenarios)

| Method | DINO-TCS↑ | SDS↑ | CLIP↑ | CCS↑ | CPS↑ |
|--------|-----------|------|-------|------|------|
| Stable Diffusion | 0.474 | 0.195 | 0.292 | -0.004 | 0.000 |
| ControlNet | 0.919 | 0.052 | 0.285 | -0.001 | 0.000 |
| AnimateDiff | 0.792 | 0.085 | 0.285 | -0.002 | 0.000 |
| **CausalDreamer** | **0.625** | **0.246** | **0.303** | **0.311** | **0.040** |

**Key Findings:**
- 4.7× better semantic diversity than ControlNet (0.246 vs 0.052)
- Only method with positive causal reasoning scores (CCS: 0.311, CPS: 0.040)
- Optimal identity-diversity tradeoff

---

## 🛠️ Model Architecture

### Causal Adapter Components

1. **Context Encoder**: 2-layer MLP (hidden dim: 1024)
2. **Delta Head**: Predicts temporal effect shift in CLIP space
3. **Attention Gates**: Query and key modulation (q_gate, k_gate)
4. **Logit Bias**: Additive attention bias for intervention

### Causal Intervention Attention

```
attn'[i,j] = q_gate[i] × attn[i,j] × k_gate[j] + exp(λ) × bias[i,j]
```

Where:
- `q_gate`, `k_gate`: Learned attention modulation
- `bias`: Causal intervention signal
- `λ`: Intervention strength (default: 0.7)

---

## 📦 Requirements

Main dependencies:

```
torch>=2.0.0
torchvision>=0.15.0
diffusers>=0.21.0
transformers>=4.30.0
accelerate>=0.20.0
pillow>=9.5.0
numpy>=1.24.0
opencv-python>=4.8.0
scikit-learn>=1.3.0
```

Full requirements in `requirements.txt`

---

## 🔧 Configuration

Edit `configs/config.yaml` to customize:

```yaml
# Model settings
clip_model: "ViT-L/14"
sd_version: "stable-diffusion-v1-5"
adapter_hidden_dim: 1024
intervention_lambda: 0.7

# Training settings
epochs: 200
batch_size: 32
learning_rate: 1.0e-4
warmup_steps: 500

# Inference settings
num_frames: 8
top_k_effects: 8
guidance_scale: 7.5
num_inference_steps: 50
```

---

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size
python train_adapter.py --batch_size 16

# Use gradient accumulation
python train_adapter.py --gradient_accumulation_steps 2
```

**2. Slow Inference**
```bash
# Reduce number of inference steps
python inference.py --num_inference_steps 25

# Use fp16 precision
python inference.py --fp16
```

**3. Poor Results**
- Check that pretrained checkpoint is loaded correctly
- Verify CLIP encoder is frozen
- Try adjusting intervention_lambda (range: 0.5-1.0)

---

## 💡 Tips for Best Results

1. **Cause Descriptions**: Use clear, action-oriented descriptions
   - ✅ Good: "a dog digging a hole in the yard"
   - ❌ Bad: "dog" or "dog is cute"

2. **Number of Frames**: 8 frames works best for most scenarios
   - Shorter sequences (4-6): Quick actions
   - Longer sequences (8-10): Gradual processes

3. **Intervention Strength**: Adjust λ based on desired identity-diversity tradeoff
   - Lower λ (0.5): Stronger identity, less variation
   - Higher λ (1.0): More variation, weaker identity

---

## 🔬 Research Paper

This code accompanies our ICME 2026 submission:

**"CausalDreamer: Temporal Image Generation through Causal Intervention Attention with Identity Preservation"**

**Abstract**: Text-to-image diffusion models generate photorealistic images but cannot produce temporally coherent sequences showing how causal events unfold while preserving subject identity. We introduce CausalDreamer, a framework for identity-preserving temporal image generation through causal intervention attention...

Full paper available upon acceptance.

---

## 📧 Contact

For questions or issues:
- Open an issue in this repository
- Contact will be provided upon acceptance

---

## 🙏 Acknowledgments

- Stable Diffusion team for the base diffusion model
- OpenAI CLIP for text-image embeddings
- DINOv2 team for identity preservation metrics
- Anonymous reviewers for valuable feedback

---

## 📜 License

Code will be released under MIT License upon acceptance.

---

## ⚠️ Note

This is an anonymous repository for double-blind review. Full author information, detailed acknowledgments, and public repository will be available upon paper acceptance.

---

**Thank you for your interest in CausalDreamer!** 🚀