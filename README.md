# PsyhookBench
## 📂Image Storage

All images related to this repository are hosted on a private **Tencent Cloud COS (Cloud Object Storage)** bucket to maintain repository performance and structure.

If you require access to the full dataset, including the bucket ID and credentials (SecretId/SecretKey), please contact the maintainer via email:

📧 **Email:** [2351470@tongji.edu.cn](mailto:2351470@tongji.edu.cn)

*Note: Access is provided for academic and research purposes only. Please include your affiliation and the purpose of your request in the email.*

## 🧪 Model Evaluation (Zero-Shot & Few-Shot)

We provide evaluation scripts for both **Zero-Shot** and **Few-Shot** testing. The implementation logic is standardized across models; therefore, we provide detailed examples using **Gemini** and **Qwen**. To test other supported models, you only need to modify the configuration section at the beginning of the respective scripts.

In the zero-shot setting, the model is tasked with classifying social media "psychological hooks" based solely on the prompt instructions without any prior examples.

* **Script:** `zero_shot_qwen.py`
* **Key Feature:** Direct inference using system prompts to evaluate the model's innate understanding of psychological triggers.

The few-shot setting provides the model with $N$ examples (exemplars) within the prompt to demonstrate the desired reasoning path and output format.

* **Script:** `few_shot_gemini.py`
* **Key Feature:** Utilizes in-context learning to improve classification accuracy for complex psychological categories like "FOMO" or "Information-gap."
