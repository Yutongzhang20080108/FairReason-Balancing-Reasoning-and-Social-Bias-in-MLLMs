# BRSS-Balancing-Reasoning-with-Social-Bias
The repo for conducting empirical study on the balance of reasoning and social bias mitigation

Training data
1. Partial Mix-of-Thoughts https://huggingface.co/datasets/Aaron080108/Partial_Mix_of_Thoughts, we randomly selectly a small subset of the orginal Mix_of_Thoughts dataset created by Open-R1 team with a size of 100k.
2. BBQ_Benchmark with reasoning traces extracted from DeepSeek-R1 https://huggingface.co/datasets/Aaron080108/BBQ_Benchmark_Reasoning_Trace, we have uploaded the cleaned data for Gender_identity, Age, Nationality, Religion and Race. For each bias catgory we selected about 1k data.


Scripts:
1. The files in extract_reasoning_data is used to extract the reasoning trace from DeepSeek-R1
2. The scripts in evaluation is used to eval any model you would like to test on either benchmarks related to reasoning, including Math and Coding or benchmarks related to bias mitigation.