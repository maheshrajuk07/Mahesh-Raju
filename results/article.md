
# Text Embeddings Benchmark Report

**Date:** 2025-12-15  
**Author:** Mahesh  
**Status:** ✅ Success

## 📋 Strategic Context
Many organizations face difficulty when deciding between high-cost third‑party APIs and operationally demanding self‑hosted embedding models. This benchmark recreates a realistic production‑like setup to identify the most effective **“Build vs. Buy”** approach, with particular attention to the balance between **inference latency costs** and **retrieval performance**.

## 🚀 Executive Summary
- **Lowest Latency Model:** `sentence-transformers/all-MiniLM-L6-v2` with an average response time of **41.05 ms**.
- **Efficiency Insight:** The donut visualizations indicate that **MiniLM** occupies the largest share in overall efficiency.
- **Deployment Guidance:** **MiniLM-L6-v2** is recommended for production systems where response speed is critical.

---

## 📊 Visualizations

### 1. Performance Overview (Bar Charts)
| Latency (Lower is Better) | Cost (Lower is Better) |
|---------------------------|------------------------|
| ![Latency](chart_bar_latency.png) | ![Cost](chart_bar_cost.png) |

---

### 2. Deep Dive Analysis (Distribution Charts)

#### A. Efficiency Score (The "Winner" Chart)
*This chart represents the performance-to-speed ratio of each model. A larger segment indicates superior efficiency.*
![Efficiency](chart_pie_efficiency.png)

#### B. Resource Consumption
| Processing Load Share | Cost Factor Share |
|-----------------------|-------------------|
| *Which model introduces the most delay?* | *Which model contributes most to cost?* |
| ![Pie Latency](chart_pie_latency.png) | ![Pie Cost](chart_pie_cost.png) |

---

## 📋 Detailed Data

| Full Name                               |   Recall@1 |   Latency (ms/req) |   Monthly Cost ($) |
|:----------------------------------------|-----------:|-------------------:|-------------------:|
| sentence-transformers/all-MiniLM-L6-v2  |       1    |              41.05 |             378.72 |
| sentence-transformers/all-mpnet-base-v2 |       0.95 |             211.73 |             378.72 |
| BAAI/bge-small-en-v1.5                  |       1    |              80.71 |             378.72 |
| BAAI/bge-base-en-v1.5                   |       1    |             217.9  |             378.72 |
| intfloat/e5-small-v2                    |       1    |              80.9  |             378.72 |

## 🧠 Analysis & Decision Matrix

| Requirement | Recommended Model | Reasoning |
|-------------|-------------------|-----------|
| **Real-time / Search** | **MiniLM-L6-v2** | Delivers the best efficiency score and operates nearly ten times faster than larger base models. |
| **Semantic Nuance** | **BGE-Base** | Offers stronger semantic depth, but accounts for more than half of the total processing load (refer Pie Chart B). |
| **Low Maintenance** | **OpenAI / API** | Requires minimal operational effort, although operating expenses increase proportionally with usage. |

## 🛠 Methodology
- **Visualization:** Charts were produced using `matplotlib` and `seaborn`, employing both bar and donut chart styles.
- **Metrics:**
    - **Efficiency Score:** Computed as `1000 / Latency`.
    - **Load Share:** Measured as the relative execution time of each model during sequential inference.

## 🔁 Reproduction
To replicate this benchmark locally:
1. Install the required packages: `pip install sentence-transformers openai pandas matplotlib seaborn`
2. Execute the benchmark script: `python run_benchmarks.py`

---

## 👨‍💻 Authors Note
This benchmark was carried out by **Mahesh** to experimentally assess the performance efficiency of locally hosted text embedding models. The results demonstrate that for high‑throughput RAG systems, the latency benefits of optimized lightweight models such as MiniLM frequently outweigh the marginal accuracy advantages of larger alternatives.
