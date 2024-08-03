# LangGPT结构化提示词编写实践

## 基础任务

- **背景问题**：近期相关研究发现，LLM在对比浮点数字时表现不佳，经验证，internlm2-chat-1.8b (internlm2-chat-7b)也存在这一问题，例如认为`13.8<13.11`。

- **任务要求**：利用LangGPT优化提示词，使LLM输出正确结果。

LangGPT 是一个帮助你编写高质量提示词的工具，理论基础是一套模块化、标准化的提示词编写方法论——结构化提示词。LangGPT (Language For GPT-like LLMs): [link](https://langgptai.feishu.cn/wiki/RXdbwRyASiShtDky381ciwFEnpe)

首先，我们使用lmdepoly部署InternLM2-Chat-1.8B模型，并调用图形化界面。初始我们设置system prompt为空。

当我们提问“13.8比13.11小吗？”，模型返回结果为：


  
