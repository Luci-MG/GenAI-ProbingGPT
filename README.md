# GenAI - Probing GPT-2 Layers for Understanding and Relationship Analysis

This repository focuses on probing and analyzing the inner workings of GPT-2, specifically looking at how its layers understand relationships between concepts, such as contradictions, neutrality, and entailments. The project is inspired by and builds on the methodology proposed in the paper *"A Structural Probe for Finding Syntax in Word Representations"* ([arXiv:1610.01644](https://arxiv.org/abs/1610.01644)).

Through this study, we aim to dissect the **GPT-2 small model** and examine how well the model learns at each layer by capturing hidden states and using classifiers to analyze relationship understanding. This repository provides tools and resources for probing GPT-2 in a structured manner, using probing techniques on various layers and performing evaluations using a supervised approach.

## Table of Contents
1. [Overview](#overview)
2. [Objectives](#objectives)
3. [Methodology](#methodology)
4. [Layer Probing and Results](#layer-probing-and-results)
5. [How to Run the Code](#how-to-run-the-code)
6. [Datasets Used](#datasets-used)
7. [Dependencies](#dependencies)
8. [Contributing](#contributing)

---

## Overview

The **GenAI - Probing GPT-2 Layers for Understanding and Relationship Analysis** project focuses on analyzing how different layers of the GPT-2 small model capture linguistic relationships such as entailment, contradiction, and neutrality between sentence pairs. This is achieved by applying **structural probing** on layers like the GPT-2 feed-forward (MLP) and attention layers.

Probing is conducted using the **SNLI (Stanford Natural Language Inference)** corpus, which is a well-known dataset for natural language inference. The goal is to evaluate how GPT-2's understanding of these relationships evolves across different layers and to identify which layers provide the best features for specific linguistic tasks.

This project is designed to be executed in **Google Colab** for ease of use, with minimal setup required. Probing is performed on specific layers of GPT-2 using the [Baukit](https://github.com/davidbau/baukit) library, and the results are analyzed by training **linear classifiers** on the extracted hidden states from each layer.

---

## Objectives

The primary objectives of this project are:

1. **Layer-wise Probing:** Understand which layers in GPT-2 contribute the most to understanding sentence-level relationships such as entailment, contradiction, and neutrality.
2. **Feature Extraction:** Extract hidden states from specific GPT-2 layers (such as MLP and Attention layers) and train linear classifiers to evaluate the quality of these features.
3. **Relationship Evaluation:** Measure how well each layer captures the semantics of contradiction, entailment, and neutral statements using the SNLI dataset.
4. **Understanding Learning Progression:** Analyze how GPT-2 learns across its layers, with particular attention to the performance of shallow layers versus deeper ones.

---

## Methodology

This project employs **structural probing**, which involves:

1. **Layer Selection:** Focusing on specific layers within GPT-2, including:
   - **Layer h.0.mlp** (one of the shallowest layers)
   - **Layer h.3.mlp**
   - **Layer h.9.mlp**
   - **Layer h.9.attn** (attention layer)

2. **Data Extraction:** Extracting the hidden states from these layers for each input pair of sentences (premise and hypothesis). The hidden states serve as feature vectors for further analysis.

3. **Training Classifiers:** A **linear classifier** is trained on the hidden states from each layer to predict the relationship between sentence pairs (entailment, contradiction, or neutrality). This allows us to evaluate how well each layer understands relationships in the input data.

4. **Evaluation and Metrics:** The performance of each layer is evaluated using standard classification metrics. Comparisons are made between different layers to determine which ones perform best at capturing the target relationships.

---

## Layer Probing and Results

### Probing Methodology

The layers are probed in a structured manner using the **Baukit** library, which allows for seamless extraction of internal representations from GPT-2. Specifically, hidden states from the following layers are probed:

- **Layer h.0.mlp:** The first feed-forward network (MLP) layer, representing very shallow understanding.
- **Layer h.3.mlp:** A middle layer, where deeper representations start to form.
- **Layer h.9.mlp:** One of the deeper MLP layers, expected to capture more abstract relationships.
- **Layer h.9.attn:** A deep attention layer, capturing contextual relationships through attention mechanisms.

### Results and Insights

After training linear classifiers on the hidden states from each of these layers, we observe the following trends:

- **Shallow Layers (h.0.mlp):** The shallow layers tend to capture basic linguistic features but struggle with complex relationships such as contradictions.
- **Middle Layers (h.3.mlp):** These layers show an improvement in relationship understanding, especially in terms of distinguishing between entailment and neutrality.
- **Deeper Layers (h.9.mlp & h.9.attn):** The deepest layers, particularly the attention layers, demonstrate the best performance in capturing contradictions and entailments. This suggests that GPT-2's deeper layers play a critical role in semantic understanding and context capture.

The detailed results of these experiments, including accuracy and error metrics, can be found in the [ProbingGPT2.ipynb](ProbingGPT2.ipynb) file.

---

## How to Run the Code

To run the probing and evaluation process:

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/GenAI-Probing-GPT-2.git
    cd GenAI-Probing-GPT-2
    ```

2. Open the **Jupyter notebook** located at:
    ```
    GenAI-Probing-GPT-2/NLPProbingGPT.ipynb
    ```

3. Follow the detailed instructions within the notebook. The entire pipeline, including data preprocessing, hidden state extraction, classifier training, and evaluation, is provided in the notebook.

4. The notebook is designed to run in **Google Colab** for seamless execution. Ensure that you have a Google Colab environment set up and GPU enabled for faster processing.

---

## Datasets Used

The primary dataset used in this project is the **SNLI (Stanford Natural Language Inference)** corpus, which contains labeled examples of sentence pairs. Each sentence pair is labeled as **Entailment**, **Contradiction**, or **Neutral**, making it ideal for evaluating relationship understanding in GPT-2.

- **SNLI Corpus**: [SNLI Dataset Paper](https://nlp.stanford.edu/pubs/snli_paper.pdf)
- **Download Link**: Available within the Jupyter notebook, or can be accessed via the Hugging Face datasets library.

---

## Dependencies

To run the code in this repository, ensure you have the following dependencies installed:

- **Transformers** (Hugging Face library for GPT-2 models)
- **Baukit** (for extracting hidden states from GPT-2)
- **Scikit-learn** (for training and evaluating linear classifiers)
- **PyTorch** (for GPU-accelerated model execution)

To install these dependencies, run:
```bash
pip install transformers baukit scikit-learn torch
```

Alternatively, these dependencies are pre-installed in **Google Colab** environments, and only minor adjustments might be needed.

---

## Contributing

We welcome contributions to the **GenAI - Probing GPT-2 Layers for Understanding and Relationship Analysis** project. If you would like to improve the code, add new features, or report issues, please feel free to submit a pull request or open an issue.

For further inquiries or feedback, reach out via email to **[Mrudhul Guda](mailto:mg6.dev@gmail.com?subject=[GitHub]%20GenAI%20Probing%20GPT-2)**.

---

This project is an exploratory dive into how GPT-2, a transformer-based language model, understands relationships across its layers. By providing a structured way to probe and analyze hidden states, this repository aims to offer insights into the inner workings of GPT-2 and how it handles complex relationships in language.