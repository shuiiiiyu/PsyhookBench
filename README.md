# PsyhookBench
## 📂Image Storage

All images related to this repository are hosted on a private **Tencent Cloud COS (Cloud Object Storage)** bucket to maintain repository performance and structure.

If you require access to the full dataset, including the bucket ID and credentials (SecretId/SecretKey), please contact the maintainer via email:

📧 **Email:** [2351470@tongji.edu.cn](mailto:2351470@tongji.edu.cn)

*Note: Access is provided for academic and research purposes only. Please include your affiliation and the purpose of your request in the email.*

## 📂 Repository Structure & Dataset Description

All data files are organized within the `dataset/` directory. The benchmark consists of human-annotated ground truth, machine-labeled data, and ethical evaluation results.

### 1. Dataset Directory (`dataset/`)

#### 🧑‍💻 Human Annotations (Expert Labels)
* **`541_raw_data.csv`**: The raw dataset containing 541 samples with metadata: `class`, `post_id`, `title`, `content`, `create_at`, `user_id`, `liked_count`, `cover_url`, `post_url`, `image_urls`, `video_url`, and `fans_count`.
* **`541_votes_reasons.csv`**: Aggregated voting data including vote counts, High-Consensus labels ($\ge 4$ votes), Edge-Case labels ($3$ votes), and qualitative audit reasons for edge cases.
* **`expert_541.csv`**: Finalized binary ground truth for the 541 samples (1 indicates presence of a hook, 0 indicates absence).
* **`expert_per.csv`**: Detailed individual voting records from the 5 expert annotators.

#### 🤖 Machine Annotations
* **`2500_raw_data.csv`**: Raw metadata for the 2,500 machine-labeled samples (same column structure as the 541 dataset).

#### 🏆 Final Combined Labels (Ground Truth)
* **`label_title_fewshot.csv`**: Combined results for Few-Shot testing. Includes both human and machine labels but excludes the 24 fixed exemplars used in prompts. (Value `2` = High Consensus, `1` = Edge Case).
* **`label_title_zeroshot.csv`**: The complete benchmark results for all 3,041 samples used in Zero-Shot testing. (Value `2` = High Consensus, `1` = Edge Case).

### 2. Ethical Evaluation (`ethics/`)
This folder contains the results of the Machine Ethics Audit.
* **`ethics_out.csv`**: Detailed ethical audit results for the 2,500 machine-labeled samples.
* **`risk_posts.csv`**: A subset of 391 samples identified by the model as potentially high-risk, flagged for manual human review.

### 3. Evaluation & Results
* **`check_csvs_fewshot/` & `check_csvs_zeroshot/`**: Reference "Answer Keys" categorized by different dimensions. These are used to compare model outputs against the gold standard.
* **`DATA_SPLIT/`**: A small-scale, balanced subset provided for initial testing and prompt engineering/tuning.
* **`raw_data_results_fewshot/` & `raw_data_results_zeroshot/`**: The raw inference outputs from various LLMs across both testing paradigms.
* **`results/`**: Comprehensive evaluation reports comparing model outputs against the finalized labels (Metrics: Accuracy, F1-score, etc.).

### 4. Data Collection
* **`xiaohongshu_download/`**: Contains the crawler scripts and tools used to fetch the original social media data from Xiaohongshu.
## 🧪 Model Evaluation (Zero-Shot & Few-Shot)

We provide evaluation scripts for both **Zero-Shot** and **Few-Shot** testing. The implementation logic is standardized across models; therefore, we provide detailed examples using **Gemini** and **Qwen**. To test other supported models, you only need to modify the configuration section at the beginning of the respective scripts.

In the zero-shot setting, the model is tasked with classifying social media "psychological hooks" based solely on the prompt instructions without any prior examples.

* **Script:** `zero_shot_qwen.py`
* **Key Feature:** Direct inference using system prompts to evaluate the model's innate understanding of psychological triggers.

The few-shot setting provides the model with $N$ examples (exemplars) within the prompt to demonstrate the desired reasoning path and output format.

* **Script:** `few_shot_gemini.py`
* **Key Feature:** Utilizes in-context learning to improve classification accuracy for complex psychological categories like "FOMO" or "Information-gap."
