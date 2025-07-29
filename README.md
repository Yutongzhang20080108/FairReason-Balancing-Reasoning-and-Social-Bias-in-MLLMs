# FairReason: Balancing Reasoning and Social Bias in MLLMs
The repo for conducting empirical study on the balance of reasoning and social bias mitigation

We provide the links to the resources we use and our final best-performance models in this research

Training data:
1. BBQ_Benchmark with reasoning traces extracted from DeepSeek-R1 https://huggingface.co/datasets/Aaron080108/BBQ_Benchmark_Reasoning_Trace, we have uploaded the cleaned data for Gender_identity, Age, Nationality, Religion and Race. For each bias catgory we selected about 1k data.
2. VLBiasBench-close-ended https://huggingface.co/datasets/Aaron080108/VLBiasBench_Close_ended, we have uploaded the original dataset to huggingface for research purposes.
3. Mix-of-Thoughts https://huggingface.co/datasets/open-r1/Mixture-of-Thoughts, a dataset used to reproduce the results for distilled models from DeepSeek.
4. LLaVA-CoT-100k https://huggingface.co/datasets/Xkev/LLaVA-CoT-100k, a Chain-of-Thoughts style of datasets used to train vision language models that can reason.

Scripts:
1. The files in extract_reasoning_data is used to extract the reasoning trace from DeepSeek-R1 and OpenAI o4-mini
2. The scripts in evaluation is used to eval any model you would like to test on either benchmarks related to reasoning, including Math and Coding or benchmarks related to bias mitigation. For MathVerse, we use the evaluation scripts from the oirginal paper https://github.com/ZrrSkywalker/MathVerse.

Final Models:
Through rigorous empirical study, we found that the best data proportion for both GRPO and model distillation is around 20%, we release the final model for further researches.
GRPO:
1. Qwen3-8B: https://huggingface.co/Aaron080108/Qwen3-8B-GRPO-20percent
2. Qwen2.5-VL-7B: https://huggingface.co/Aaron080108/Qwen2.5-VL-7B-GRPO-20percent
Model Distillation
1. Qwen3-8B: https://huggingface.co/Aaron080108/Qwen3-8B-Distillation-20percent
2. Qwen2.5-VL-7B: https://huggingface.co/Aaron080108/Qwen2.5-VL-7B-Distillation-20percent