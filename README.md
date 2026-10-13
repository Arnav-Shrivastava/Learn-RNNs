# RNN, LSTM and GRU Sequences

A comprehensive, zero-to-hero learning curriculum covering Recurrent Neural Networks (RNNs), Long Short-Term Memory networks (LSTMs), and Gated Recurrent Units (GRUs) implemented in NumPy and PyTorch.

## Overview
This repository contains interactive, fully commented notebooks covering:
1. **Sequence Modeling & BPTT Theory**: Intuition, hidden states, hand-worked 3-step BPTT, vanishing/exploding gradients.
2. **Vanilla RNN Cell**: Scratch implementation in NumPy and PyTorch verification.
3. **Vanishing & Exploding Gradients**: Empirical demonstrations across sequence lengths (10, 50, 200) and gradient clipping.
4. **Gated Architectures (LSTM & GRU)**: Gates from scratch, cell states, PyTorch verification, and comparison.
5. **Part A - Character-Level Text Generation**: Char-level LSTM trained on text, sampling at epochs 1, 10, 50, and temperature sampling.
6. **Part B - Streaming Time-Series & Anomaly Detection**: Real-time hidden state stepping vs. full-window recomputation latency, anomaly detection.
7. **Long-Range Memory Benchmark**: RNN vs. LSTM vs. Transformer on synthetic copy/memory tasks.
8. **Modern Architecture Decision Guide**: Trade-offs between RNN/LSTM, Transformers, State Space Models (Mamba), and when to use each.
9. **Mastery Challenge**: 8 "prove it" questions, 3 coding exercises, and 1 capstone extension idea.

## Repository Structure
```text
├── notebooks/
├── samples/
├── assets/
├── requirements.txt
├── .gitignore
└── README.md
```

## Quick Start
```bash
pip install -r requirements.txt
```
