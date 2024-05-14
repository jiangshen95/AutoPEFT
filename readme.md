Hello AutoPEFT!

## 开发计划

- 基本微调框架实现
  - 基于已有框架，实现高效且灵活的微调框架，可以用于各种模型和 PEFT 方法
  - 进一步实现不同 PEFT 方法的组合
- 剪枝模块实现
  - 实现和提供各种剪枝方法，剪枝 PEFT 组合，并验证剪枝的效果
- 验证已有实现的效果，尤其是在大模型上的效果
- 实现架构搜索模块

## TODO
- 构建评估 Benchmark
- prefix-tuning
- verbose
- baseline：普通PEFT，比如LORA
- 没做剪枝的baseline
- 使用AUC
- 物理结构剪枝
- debug训练

## 评估

LLM 微调性能评估 Benchmark 构建，包含 NLG (Neural language generation) 和 NLU (Neural language understanding) 任务。

- 任务：指令微调、Summarization、机器翻译等
- 数据集：Alpaca 等
- 评估度量：MT-bench 和 SuperGLUE 等。

可能使用的数据集: GLUE, super-GLUE, 
- BigBench https://github.com/google/BIG-bench 谷歌的大模型的测试基准
- Commonsense Reasoning Benchmark https://commonsense.run/datasets/ 常识推理
- Squad2.0 https://rajpurkar.github.io/SQuAD-explorer/explore/v2.0/dev/ 问答
- MBPP, HumanEval 代码生成
