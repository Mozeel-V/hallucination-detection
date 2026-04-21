# SMILES-2026 Hallucination Detection: Geometric & Semantic Probing

## 1. Reproducibility Instructions

**Environment Setup & Requirements:**
This solution was developed and tested on an environment equipped with an NVIDIA GPU. In addition to the base dependencies, `scikit-learn` is required for dimensionality reduction and cross-validation splitting.

Run the following commands to set up the environment and execute the pipeline:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install scikit-learn pandas numpy
python solution.py
```

**Implementation Details to Reproduce Results:**
The pipeline relies on custom geometric feature extraction. To ensure the complete feature set is utilized during evaluation, the `USE_GEOMETRIC` flag inside the fixed `solution.py` file must be set to `True`. 

```python
USE_GEOMETRIC = True
```
Running `solution.py` will automatically parse the dataset, extract the specified hidden states, compute the geometric trajectories, train the MLP probe using 5-Fold Stratified Cross-Validation, and generate both `results.json` and the final `predictions.csv`.

---

## 2. Final Solution Description

My approach moves beyond traditional linear probing of mean-pooled embeddings. Instead, it integrates **Topological/Geometric Features** with **Semantic (Hidden State) Features**, which are then compressed and evaluated via a non-linear neural probe. 

### What Components Were Modified?
1. **`aggregation.py`**: Completely rewritten to target specific middle-to-late transformer layers and to extract advanced geometric trajectories (EigenScores and LSD convergence metrics).
2. **`probe.py`**: Upgraded from a simple linear classifier to a Scikit-Learn Pipeline combining PCA dimensionality reduction with a lightweight, heavily regularized Multi-Layer Perceptron (MLP) built in PyTorch.
3. **`splitting.py`**: Replaced the static split with a 5-Fold Stratified Cross-Validation routine to ensure robust threshold tuning and metric evaluation.
4. **`solution.py`**: Toggled the `USE_GEOMETRIC` flag to `True`.

### Final Approach and Theoretical Grounding

**A. Layer Selection (The Factual Recall Zone)**
Instead of extracting all 24 layers, the probe strictly targets **layers 12 through 20**. 
* *Why?* Mechanistic interpretability research demonstrates that factual recall acts as a key-value memory localized in the middle layers of Transformer architectures (Geva et al., EMNLP 2021). Early layers primarily handle shallow lexical patterns and entity resolution, while late layers focus on formatting and grammar. Furthermore, an ICLR 2025 analysis (Sujan et al.) found that truthfulness signals are heavily distributed across the middle layers, with the absolute final layer being non-critical for factual probing.

**B. Semantic and Geometric Feature Extraction (`aggregation.py`)**
I extracted the hidden state of the final non-padding token. To enrich this representation, I implemented two State-of-the-Art geometric methodologies:
1. **INSIDE / EigenScore (Chen et al., ICLR 2024):** Hallucinations cause the model's internal states to geometrically scatter. I computed the Gram matrix (a proxy for the covariance matrix) of the hidden states across the target layers. Extracting the top 3 eigenvalues of this matrix provides a robust "EigenScore" that quantifies the geometric stability of the model's internal representation.
2. **Layer-wise Semantic Dynamics / LSD (Amir-Hameed et al., 2024):** Factual statements maintain monotonic alignment across layers, while hallucinated content exhibits erratic drift. I quantified this trajectory by calculating the *Convergence* (cosine similarity of each intermediate layer against the final target layer) and *Curvature/Drift* (the L2 step distance between consecutive layers). 

**C. Dimensionality Reduction & Non-Linear Classification (`probe.py`)**
Concatenating raw hidden states across 9 layers results in a massively high-dimensional vector (~8000+ dimensions). Training directly on this with only 689 samples leads to the curse of dimensionality.
* **PCA:** I utilized Principal Component Analysis to compress the scaled semantic vectors down to 64 principal components, effectively capturing the variance without the noise.
* **Lightweight MLP:** Linear probes often fail to capture the complex geometry of fabricated information in deep semantic spaces. I implemented a highly regularized MLP (utilizing Dropout, BatchNorm1d, and AdamW with weight decay) combined with a `BCEWithLogitsLoss` weighted to handle any class imbalance.

---

## 3. Experiments and Failed Attempts

**What ideas were tried but discarded?**

* **Laplacian Eigenvalues of Attention Maps (LapEigvals):** Inspired by Binkowski et al. (EMNLP 2025), I initially attempted to treat the model's attention maps as graph adjacency matrices to compute spectral features. *Reason for discarding:* The fixed evaluation infrastructure (`solution.py`) invokes the model without the `output_attentions=True` flag. Modifying the fixed infrastructure to extract attention weights would violate the reproducibility constraints. This led directly to the successful pivot toward the LSD and INSIDE methods, which achieve similar geometric insights relying exclusively on the permitted `hidden_states`.
* **All-Layer Concatenation:** I initially concatenated the hidden states from all 24 layers. *Reason for discarding:* This caused severe overfitting on the training split (achieving near-perfect accuracy but poor validation F1). The noise from the early grammatical layers and late formatting layers drowned out the factual signals present in the middle layers.
* **Logistic Regression / Linear Probing:** I attempted to use a standard Logistic Regression classifier on the concatenated embeddings. *Reason for discarding:* The data exhibited non-linear separability. Factual drift in the semantic space is highly complex; the linear probe struggled to separate the classes cleanly compared to the lightweight MLP.
* **Mean-Pooling Prompt Tokens:** I experimented with mean-pooling the hidden states of the entire prompt alongside the generated response. *Reason for discarding:* The prompt tokens merely represent the user's question, not the model's internal confidence in its fabricated answer. Including them diluted the hallucination signal.

--- 

### References
* Geva, M., et al. (2021). Transformer Feed-Forward Layers Are Key-Value Memories. *EMNLP*.
* Chen, C., et al. (2024). INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection. *ICLR*.
* Amir-Hameed, M., et al. (2024). Layer-wise Semantic Dynamics (LSD): A geometric framework for hallucination detection.
* Sujan, et al. (2025). Truthfulness in LLMs: A Layer-wise Comparative Analysis. *ICLR Workshop*.
