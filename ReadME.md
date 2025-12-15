
# Technical Report: Text Embeddings Benchmark

**Author:** Mahesh  
**Date:** 2025-12-15  
**Context:** AI/ML Intern Assignment (Process Point Technologies)  
**Status:** ✅ Complete

---

## 1. Abstract
This work presents a structured benchmarking system designed to compare the performance characteristics of modern NLP embedding models. As AI systems move from experimentation to real-world deployment, selecting an appropriate embedding model becomes critical due to its direct impact on system latency and infrastructure expenditure. In this benchmark, three models were systematically evaluated: **MiniLM-L6-v2** (Distilled), **BGE-Small** (Compact), and **BGE-Base** (Standard).

The experimental findings clearly indicate that although larger models may demonstrate marginal advantages on academic benchmarks, the **MiniLM model achieves nearly a 79% reduction in inference time** relative to the Base model (38ms versus 185ms), while still maintaining **perfect Recall@1** on a controlled synthetic evaluation set. This balance of speed and accuracy makes MiniLM highly suitable for CPU-based production environments where operational cost efficiency is essential. By validating performance on structured synthetic data, this study demonstrates that distilled models are well-suited for Retrieval Augmented Generation (RAG) systems without incurring the computational cost associated with larger architectures.

## 2. Introduction
Within modern AI pipelines, text embeddings function as a core component for semantic search, document similarity, clustering, and RAG-based systems. Unlike traditional lexical retrieval approaches such as BM25, embedding models transform textual input into dense numerical vectors that encode semantic meaning and contextual intent.

However, transitioning these systems into production introduces several practical trade-offs that are often underestimated during early experimentation. Development teams must decide between external API-based solutions (e.g., OpenAI or Cohere), which provide rapid integration but incur usage-based costs, and self-hosted open-source models, which require infrastructure management but offer predictable expenses and improved data privacy.

The central aim of this assignment was to construct a practical **Build vs. Buy evaluation framework**. By benchmarking models across three primary dimensions—**Latency**, **Retrieval Quality**, and **Total Cost of Ownership (TCO)**—this analysis provides actionable insights for real-world deployment decisions. Rather than focusing solely on accuracy, this report emphasizes operational performance and illustrates how architectural design choices directly influence inference speed and user experience.

## 3. Methodology

### 3.1 Synthetic Dataset Generation
To ensure reproducibility and avoid reliance on large external datasets, a **Synthetic Dataset Generator** was implemented. This generator produces a controlled corpus consisting of 50 documents spanning multiple knowledge domains such as Physics, Biology, and History. Additionally, 20 query samples were programmatically generated, each mapped to a specific ground-truth document through explicit semantic relationships.

This controlled setup removes the ambiguity and noise commonly present in large-scale public datasets like MSMARCO. As a result, retrieval failures can be attributed solely to model limitations rather than labeling inconsistencies or unclear semantic mappings. This design confirms whether each model can reliably capture and align semantic intent under ideal conditions.

### 3.2 Code Implementation Strategy
The benchmarking pipeline was developed in Python using a modular wrapper-based architecture, allowing seamless integration of additional models in the future. A crucial part of the evaluation process is the **Latency Warmup Mechanism**. When running transformer models locally via PyTorch and HuggingFace, the initial inference is typically slower due to memory allocation and internal optimization steps.

To prevent this cold-start overhead from distorting performance metrics, each model undergoes a warmup phase before latency measurements begin. This ensures that recorded timings represent steady-state performance, which is far more relevant for production environments. The following snippet illustrates the measurement approach:

```python
def benchmark_latency(model_wrapper, texts, runs=5):
    # Warmup Phase: Critical for fair local comparison.
    # Forces weights into RAM and JIT compilation.
    print(f"  ...Warming up {model_wrapper.name}...")
    model_wrapper.encode(["warmup"] * 2)

    latencies = []
    for _ in range(runs):
        start = time.time()
        model_wrapper.encode(texts) # Actual Inference
        latencies.append((time.time() - start) * 1000)

    return np.mean(latencies)
```

## 4. Results & Analysis
All benchmarks were executed in a local environment to simulate an edge or on-premise deployment scenario. Each evaluated model achieved a **Recall@1 score of 1.0**, successfully retrieving the correct document for every query in the synthetic dataset. This outcome suggests that for well-defined semantic tasks, smaller models are sufficiently capable of accurate retrieval.

### 4.1 Summary Table
The table below presents a consolidated overview of retrieval quality, latency, and estimated operational cost (based on AWS g4dn.xlarge hosting).

| Model Name | Recall@1 | Latency (Mean) | Est. Monthly Cost |
|:---|:---:|:---:|:---:|
| **MiniLM-L6-v2** | 1.0 | **38.0 ms** | ~$378 (Fixed) |
| **BGE-Small-en** | 1.0 | 70.4 ms | ~$378 (Fixed) |
| **BGE-Base-en** | 1.0 | 185.0 ms | ~$378 (Fixed) |

### 4.2 Latency and Throughput Analysis
Inference speed is a key determinant of user satisfaction in interactive systems such as chatbots and search interfaces. Delays exceeding 100ms are often noticeable to users. As shown in Figure 1, inference latency increases proportionally with model size. **MiniLM-L6-v2** demonstrates an average latency of **38.0 ms**, making it approximately **4.8 times faster** than the BGE-Base model.

This improvement stems from architectural efficiency, including fewer transformer layers and reduced hidden dimensions. In high-throughput environments handling hundreds of requests per second, adopting MiniLM can significantly increase system capacity without additional hardware investment.

![Latency Chart](results/chart_bar_latency.png)

### 4.3 Cost Efficiency and Value
To quantify efficiency, an **Efficiency Score** was defined as `1000 / Latency`, capturing the trade-off between speed and resource usage. Under this metric, MiniLM clearly outperforms the other models. Hosting these models locally on an AWS `g4dn.xlarge` instance results in a fixed monthly cost of approximately **$378**.

Although API-based services may appear cost-effective at low usage volumes, their expenses scale linearly with demand. At higher throughput levels, self-hosted solutions quickly become more economical. In this context, MiniLM provides the optimal balance between performance and cost for enterprise-scale workloads.

![Efficiency Chart](results/chart_pie_efficiency.png)

### 4.4 Cost Comparison Visualization
The visualization below compares estimated monthly costs across deployment options. Local model expenses remain constant due to fixed infrastructure requirements, whereas API-based costs fluctuate based on query volume.

![Cost Chart](results/chart_bar_cost.png)

## 5. Conclusion & Recommendations
This benchmark reinforces a key engineering principle: **larger models are not always the most practical choice**. While larger architectures may achieve higher scores on complex academic benchmarks, many real-world applications prioritize responsiveness and efficiency.

The **MiniLM-L6-v2** model demonstrated excellent retrieval performance with significantly lower latency, making it well-suited for production use cases. Its ability to deliver accurate results in under 50ms positions it as an ideal solution for most semantic retrieval workloads.

### Final Recommendation
For production deployment of internal knowledge search systems, RAG pipelines, and document similarity services, **MiniLM-L6-v2** is strongly recommended. It satisfies the performance needs of the majority of applications while minimizing infrastructure cost and maximizing responsiveness. Larger models such as BGE-Base should be reserved for offline or batch-processing scenarios where deeper semantic representation is required and latency constraints are relaxed.

---

## 6. How to Reproduce This Benchmark
### Hardware Requirements
* **CPU:** Dual Core 2.0GHz or better (AVX support recommended)
* **RAM:** 4GB Minimum
* **Storage:** 2GB Free space

### Quick Start
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install sentence-transformers openai pandas matplotlib seaborn pyyaml tabulate
   ```
3. Run the benchmark:
   ```bash
   python run_benchmarks.py
   ```
