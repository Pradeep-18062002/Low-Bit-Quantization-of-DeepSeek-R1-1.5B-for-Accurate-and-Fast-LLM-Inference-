# Low-Bit-Quantization-of-DeepSeek-R1-for-Accurate-and-Fast-LLM-Inference-

# DeepSeek Lite: Optimizing DeepSeek R1 with Quantization

A comparative study on compressing DeepSeek-R1 using advanced post-training quantization techniques such as SmoothQuant, QuantizationModifier, and GPTQModifier, evaluated on benchmarks like MMLU and GSM8K.

---

## Overview

Large Language Models (LLMs) like DeepSeek-R1 offer strong performance across reasoning, programming, and math benchmarks. However, their massive size (1.5B and 7B parameters) makes them expensive to deploy in real-time or on edge devices.

This project focuses on using LLMCompressor, a Python-native quantization toolkit, to reduce model size and inference cost via:
- SmoothQuant – smooths activations for better quantization
- QuantizationModifier – fast, rule-based quantization
- GPTQModifier – Hessian-aware, precision-preserving quantization

---

## Goals

- Compress DeepSeek-R1 (1.5B and 7B) to formats like W4A16 and W8A8
- Maintain high accuracy on MMLU (High School CS) and GSM8K
- Benchmark inference time and memory footprint
- Recommend quantization formats for deployment

---

## Techniques Used

| Technique               | Description |
|------------------------|-------------|
| SmoothQuant            | Reduces outliers in activations before quantization |
| QuantizationModifier   | Applies static quantization rules (W8A8, W4A16) |
| GPTQModifier           | Greedy Hessian-based PTQ that preserves important weights |
| LLMCompressor          | Framework to apply quantization to HuggingFace models |

---

## Datasets

- MMLU – High School Computer Science  
  Tests programming, data structures, logic, and algorithms

- GSM8K – Grade School Math 8K  
  Math word problems for arithmetic and step-by-step reasoning

---

## Experiments

We compared DeepSeek-R1 under 4 configurations:
- W8A8 + QuantizationModifier
- W4A16 + QuantizationModifier
- W8A8 + GPTQModifier
- W4A16 + GPTQModifier

### Metrics:
- Classification accuracy (MMLU)
- Exact match score (GSM8K)
- Inference latency (seconds)
- GPU memory usage (MB)

---

## Results Summary

| Format     | Accuracy (1.5B) | Accuracy (7B) | Inference Time | GPU Memory |
|------------|------------------|---------------|----------------|-------------|
| Float32    | Baseline          | Baseline       | 4.17 sec       | 7015 MB     |
| W8A8       | ~90-95%           | ~97%           | 2.91 sec       | 3511 MB     |
| W4A16      | ~80-85%           | ~90-92%        | 7.05 sec       | 4109 MB     |

Conclusion: W8A8 offers the best trade-off. W4A16 works with GPTQ, but is hardware-sensitive.

---

## Key Takeaways

- SmoothQuant + GPTQModifier preserves accuracy even with INT4
- W8A8 format is the best for deployment on most GPUs
- Larger models (7B) are more robust to quantization than smaller ones (1.5B)
- Quantization is not just for compression — it's essential for latency and cost-sensitive inference

---

## Team

- Pradeep Raj Prabhu Raj – GPTQModifier pipeline, result analysis, report writing  
- Subha Ilamathy – QuantizationModifier pipeline, benchmarking, evaluation

---

## Future Work

- Compare with FP8 quantization and 2:4 structured sparsity
- Add SparseGPT and mixed-precision configurations
- Extend evaluations to real deployment scenarios (mobile, web inference)

---

## References

- SmoothQuant: https://arxiv.org/abs/2306.00978  
- GPTQ: https://arxiv.org/abs/2210.17323  
- MMLU Benchmark: https://arxiv.org/abs/2009.03300  
- GSM8K: https://arxiv.org/abs/2110.14168  
