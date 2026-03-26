# From-Scratch Transformer with Mixture-of-Experts: A Comparative Study

A decoder-only Transformer language model built entirely from scratch in PyTorch, implementing the core components of the modern LLaMA architecture. Extended with a Mixture-of-Experts (MoE) layer, then evaluated through a controlled 4-way comparison studying the effects of sparse expert routing and dropout regularization.

## Architecture

All components are implemented from scratch with no external model libraries:

| Component | Implementation |
|---|---|
| Positional Encoding | Rotary Position Embeddings (RoPE) via complex-number rotation |
| Normalization | RMSNorm (pre-norm, as in LLaMA) |
| Attention | Multi-head self-attention with causal masking |
| Feed-Forward | SwiGLU activation (gate + content paths) |
| Expert Routing | Top-2 gating across 4 experts with softmax-based router |
| Load Balancing | Auxiliary loss penalizing uneven expert utilization |
| Regularization | Configurable dropout on attention weights and FFN output |

## Experimental Setup

Trained and evaluated **4 model variants** on a character-level Shakespeare dataset (~1.1M characters, 65-token vocabulary):

| Model | Total Params | Active Params/Token | Dropout | Weight Decay |
|---|---|---|---|---|
| MoE | 2.44M | 1.36M (55.8% active) | 0.0 | 0.01 |
| FFN (Dense) | 0.82M | 0.82M (100% active) | 0.0 | 0.01 |
| MoE + Dropout | 2.44M | 1.36M (55.8% active) | 0.1 | 0.1 |
| FFN + Dropout | 0.82M | 0.82M (100% active) | 0.1 | 0.1 |

## Results

| Metric | MoE | FFN | MoE + Dropout | FFN + Dropout |
|---|---|---|---|---|
| Validation Loss | 7.742 | 4.440 | 2.364 | **1.665** |
| Perplexity | 2303 | 84.73 | 10.64 | **5.29** |
| Top-1 Accuracy | — | 33.48% | 53.16% | **55.53%** |
| Top-5 Accuracy | — | 62.81% | 81.73% | **84.04%** |

## Key Findings

1. **MoE without regularization severely overfits** — achieved the lowest training loss but the worst validation perplexity (2303), performing worse than random guessing on held-out data. Diagnosed via train/val loss divergence.
2. **Dropout is critical for MoE at small data scale** — adding dropout + weight decay reduced MoE's perplexity from 2303 to 10.64, a 99.5% improvement.
3. **Expert load balancing worked as intended** — the auxiliary loss maintained near-uniform token distribution (~25% per expert) across all layers, preventing expert collapse.
4. **Dense FFN + Dropout wins at small scale** — the regularized dense model achieved the best perplexity (5.29) and accuracy (55.53% top-1), demonstrating that MoE's benefits depend on sufficient data scale, consistent with findings from Mixtral, Switch Transformer, and GShard.

## Evaluation Metrics

- **Perplexity** — standard language model evaluation metric
- **Top-1 / Top-5 accuracy** — next-token prediction accuracy
- **Expert load balancing heatmaps** — token distribution across experts per layer
- **Parameter efficiency analysis** — total vs. active parameters, sparsity ratio

## Project Structure

```
.
├── README.md
├── requirements.txt
├── tiny_shakespeare.ipynb          # Full implementation, training, and evaluation
├── shakespare/
│   └── input.txt          # Shakespeare dataset (~1.1M characters)
└── checkpoints/           # Saved model weights (not tracked in git)
```

## Getting Started

```bash
git clone https://github.com/<your-username>/transformer-moe-study.git
cd transformer-moe-study
pip install -r requirements.txt
```

Open `tiny_shakespeare.ipynb` and run all cells. To skip training and use pre-trained checkpoints, run cells 0–34 (model definitions + dataset), then cell 40 (load checkpoints), then cells 41+ (generation and evaluation).

## Technologies

Python, PyTorch, Jupyter Notebook, Matplotlib
