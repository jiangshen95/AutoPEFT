Hello AutoPEFT!

## 开发计划

- 基本微调框架实现
  - 基于已有框架，实现高效且灵活的微调框架，可以用于各种模型和 PEFT 方法
  - 进一步实现不同 PEFT 方法的组合
- 剪枝模块实现
  - 实现和提供各种剪枝方法，剪枝 PEFT 组合，并验证剪枝的效果
- 验证已有实现的效果，尤其是在大模型上的效果
- 实现架构搜索模块

- 把找名字函数放到pruning里
- 剪枝函数放到pruning里
- 在剪枝里放不同PEFT的实现
## TODO
- baseline：普通PEFT，比如LORA
- 没做剪枝的baseline
- 使用AUC
- 物理结构剪枝
- debug训练