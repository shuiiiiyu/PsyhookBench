# PsyhookBench
## Appendix

Our work has been submitted. For more comprehensive details, supplementary analyses, and extended experimental results, please refer to the [Appendix.pdf](./Appendix.pdf) available in this repository.

The **Appendix.pdf** includes the following peer-reviewed supplementary sections:

#### 1. Detailed Operational Definitions
A deep dive into the psychological theories (e.g., **McGuire’s motivational matrix**, **FOMO**, **Information-gap**) and the specific operationalization criteria used for human and model annotation.
#### 2.Invalid Output Analysis
#### 3.Performance Different Annotators
#### 4.Annotation Platform Interface
#### 5.Model Performance Across Vertical Categories
#### 6.Statistical Characterization of Psychological Hooks
#### 7.Dataset Composition and Construction Details
#### 8.Detailed Related Work Supplement
#### 9.Definitions of Ethical Risks Detail
## Dataset Image Storage

All images related to this repository are hosted on a private **Tencent Cloud COS (Cloud Object Storage)** bucket to maintain repository performance and structure.

PsyHookBench is a multimodal benchmark containing over **30GB** of social media data, specifically designed to evaluate how Vision-Language Models (VLMs) understand psychological persuasion ("hooks").

We provide two ways to interact with the dataset, depending on your needs:

1.  **Interactive Explorer (Recommended)** [http://124.221.85.147:5000](http://124.221.85.147:5000)  
    *Browse the image-label pairs directly in your browser without downloading anything.*

2.  **Full Dataset Download (ZIP)** [Download full dataset image](https://shuiiiiyu-1390064103.cos.ap-shanghai.myqcloud.com/dataset-img.zip)  
    *Direct high-speed download from Tencent Cloud COS.*
3.  **Complete Annotations (CSV):** [Download raw data and labeled data](https://shuiiiiyu-1390064103.cos.ap-shanghai.myqcloud.com/dataset_csv.zip))  
    *The full ground-truth file mapping `post_id` to psychological hook scores (0-2).*
---

## Dataset Structure & Labels

Our dataset categorizes psychological hooks into 8 core mechanisms based on social psychology:

| ID | Hook Mechanism | Description |
| :--- | :--- | :--- |
| 1 | **FOMO** | Fear of Missing Out / Loss Aversion |
| 2 | **Gain Appeal** | Emphasizing benefits, efficiency, or rewards |
| 3 | **Information-gap** | Creating curiosity through intentional omission |
| 4 | **Anomaly & Novelty** | Counterintuitive or rare content |
| 5 | **Perceptual Contrast** | Visual or textual "Before vs. After" |
| 6 | **Ingroup/Outgroup** | Social identity and group belonging |
| 7 | **Social Comparison** | Relative status, gaps, and competition |
| 8 | **Authority** | Expert endorsements and certifications |
      
## Repository Structure & Dataset Description

All data files are organized within the `dataset/` directory. The benchmark consists of human-annotated ground truth, machine-labeled data, and ethical evaluation results.

### 1. Dataset Directory (`dataset/`)

####  Human Annotations (Expert Labels)
* **`541_raw_data.csv`**: The raw dataset containing 541 samples with metadata: `class`, `post_id`, `title`, `content`, `create_at`, `user_id`, `liked_count`, `cover_url`, `post_url`, `image_urls`, `video_url`, and `fans_count`.
* **`541_votes_reasons.csv`**: Aggregated voting data including vote counts, High-Consensus labels ($\ge 4$ votes), Edge-Case labels ($3$ votes), and qualitative audit reasons for edge cases.
* **`expert_541.csv`**: Finalized binary ground truth for the 541 samples (1 indicates presence of a hook, 0 indicates absence).
* **`expert_per.csv`**: Detailed individual voting records from the 5 expert annotators.

####  Machine Annotations
* **`2500_raw_data.csv`**: Raw metadata for the 2,500 machine-labeled samples (same column structure as the 541 dataset).

####  Final Combined Labels (Ground Truth)
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
##  Model Evaluation (Zero-Shot & Few-Shot)

We provide evaluation scripts for both **Zero-Shot** and **Few-Shot** testing. The implementation logic is standardized across models; therefore, we provide detailed examples using **Gemini** and **Qwen**. To test other supported models, you only need to modify the configuration section at the beginning of the respective scripts.

In the zero-shot setting, the model is tasked with classifying social media "psychological hooks" based solely on the prompt instructions without any prior examples.

* **Script:** `zero_shot_qwen.py`
* **Key Feature:** Direct inference using system prompts to evaluate the model's innate understanding of psychological triggers.

The few-shot setting provides the model with $N$ examples (exemplars) within the prompt to demonstrate the desired reasoning path and output format.

* **Script:** `few_shot_gemini.py`
* **Key Feature:** Utilizes in-context learning to improve classification accuracy for complex psychological categories like "FOMO" or "Information-gap."
