## 为什么要研究大模型的评测？

1. 首先，研究评测对于我们全面了解大型语言模型的优势和限制至关重要。尽管许多研究表明大型语言模型在多个通用任务上已经达到或超越了人类水平，但仍然存在质疑，即这些模型的能力是否只是对训练数据的记忆而非真正的理解。例如，即使只提供LeetCode题目编号而不提供具体信息，大型语言模型也能够正确输出答案，这暗示着训练数据可能存在污染现象。
2. 其次，研究评测有助于指导和改进人类与大型语言模型之间的协同交互。考虑到大型语言模型的最终服务对象是人类，为了更好地设计人机交互的新范式，我们有必要全面评估模型的各项能力。
3. 最后，研究评测可以帮助我们更好地规划大型语言模型未来的发展，并预防未知和潜在的风险。随着大型语言模型的不断演进，其能力也在不断增强。通过合理科学的评测机制，我们能够从进化的角度评估模型的能力，并提前预测潜在的风险，这是至关重要的研究内容。
4. 对于大多数人来说，大型语言模型可能似乎与他们无关，因为训练这样的模型成本较高。然而，就像飞机的制造一样，尽管成本高昂，但一旦制造完成，大家使用的机会就会非常频繁。因此，了解不同语言模型之间的性能、舒适性和安全性，能够帮助人们更好地选择适合的模型，这对于研究人员和产品开发者而言同样具有重要意义。

## OpenCompass介绍

上海人工智能实验室科学家团队正式发布了大模型开源开放评测体系 “司南” (OpenCompass2.0)，用于为大语言模型、多模态模型等提供一站式评测服务。其主要特点如下：

开源可复现：提供公平、公开、可复现的大模型评测方案

全面的能力维度：五大维度设计，提供 70+ 个数据集约 40 万题的的模型评测方案，全面评估模型能力

丰富的模型支持：已支持 20+ HuggingFace 及 API 模型

分布式高效评测：一行命令实现任务分割和分布式评测，数小时即可完成千亿模型全量评测

多样化评测范式：支持零样本、小样本及思维链评测，结合标准型或对话型提示词模板，轻松激发各种模型最大性能

灵活化拓展：想增加新模型或数据集？想要自定义更高级的任务分割策略，甚至接入新的集群管理系统？OpenCompass 的一切均可轻松扩展！


## 评测对象

本算法库的主要评测对象为语言大模型与多模态大模型。我们以语言大模型为例介绍评测的具体模型类型。

基座模型：一般是经过海量的文本数据以自监督学习的方式进行训练获得的模型（如OpenAI的GPT-3，Meta的LLaMA），往往具有强大的文字续写能力。
对话模型：一般是在的基座模型的基础上，经过指令微调或人类偏好对齐获得的模型（如OpenAI的ChatGPT、上海人工智能实验室的书生·浦语），能理解人类指令，具有较强的对话能力。


## 工具架构

<img src="tool.png" alt="Resized Image 1" width="800"/>

模型层：大模型评测所涉及的主要模型种类，OpenCompass 以基座模型和对话模型作为重点评测对象。

能力层：OpenCompass 从本方案从通用能力和特色能力两个方面来进行评测维度设计。在模型通用能力方面，从语言、知识、理解、推理、安全等多个能力维度进行评测。在特色能力方面，从长文本、代码、工具、知识增强等维度进行评测。

方法层：OpenCompass 采用客观评测与主观评测两种评测方式。客观评测能便捷地评估模型在具有确定答案（如选择，填空，封闭式问答等）的任务上的能力，主观评测能评估用户对模型回复的真实满意度，OpenCompass 采用基于模型辅助的主观评测和基于人类反馈的主观评测两种方式。

工具层：OpenCompass 提供丰富的功能支持自动化地开展大语言模型的高效评测。包括分布式评测技术，提示词工程，对接评测数据库，评测榜单发布，评测报告生成等诸多功能。

## 设计思路

为准确、全面、系统化地评估大语言模型的能力，OpenCompass 从通用人工智能的角度出发，结合学术界的前沿进展和工业界的最佳实践，提出一套面向实际应用的模型能力评价体系。OpenCompass 能力维度体系涵盖通用能力和特色能力两大部分。

## 评测方法

OpenCompass 采取客观评测与主观评测相结合的方法。针对具有确定性答案的能力维度和场景，通过构造丰富完善的评测集，对模型能力进行综合评价。针对体现模型能力的开放式或半开放式的问题、模型安全问题等，采用主客观相结合的评测方式。

### 客观评测

针对具有标准答案的客观问题，我们可以通过使用定量指标比较模型的输出与标准答案的差异，并根据结果衡量模型的性能。同时，由于大语言模型输出自由度较高，在评测阶段，我们需要对其输入和输出作一定的规范和设计，尽可能减少噪声输出在评测阶段的影响，才能对模型的能力有更加完整和客观的评价。 为了更好地激发出模型在题目测试领域的能力，并引导模型按照一定的模板输出答案，OpenCompass 采用提示词工程 （prompt engineering）和语境学习（in-context learning）进行客观评测。 在客观评测的具体实践中，我们通常采用下列两种方式进行模型输出结果的评测：

判别式评测：该评测方式基于将问题与候选答案组合在一起，计算模型在所有组合上的困惑度（perplexity），并选择困惑度最小的答案作为模型的最终输出。例如，若模型在答案1上的困惑度为 0.1，在答案2 上的困惑度为 0.2，最终我们会选择 答案1 作为模型的输出。

生成式评测：该评测方式主要用于生成类任务，如语言翻译、程序生成、逻辑分析题等。具体实践时，使用问题作为模型的原始输入，并留白答案区域待模型进行后续补全。我们通常还需要对其输出进行后处理，以保证输出满足数据集的要求。

### 主观评测

语言表达生动精彩，变化丰富，大量的场景和能力无法凭借客观指标进行评测。针对如模型安全和模型语言能力的评测，以人的主观感受为主的评测更能体现模型的真实能力，并更符合大模型的实际使用场景。 OpenCompass 采取的主观评测方案是指借助受试者的主观判断对具有对话能力的大语言模型进行能力评测。在具体实践中，我们提前基于模型的能力维度构建主观测试问题集合，并将不同模型对于同一问题的不同回复展现给受试者，收集受试者基于主观感受的评分。由于主观测试成本高昂，本方案同时也采用使用性能优异的大语言模拟人类进行主观打分。在实际评测中，本文将采用真实人类专家的主观评测与基于模型打分的主观评测相结合的方式开展模型能力评估。 在具体开展主观评测时，OpenComapss 采用单模型回复满意度统计和多模型满意度比较两种方式开展具体的评测工作。

## OpenCompass评测快速开始

### 概览

在 OpenCompass 中评估一个模型通常包括以下几个阶段：配置 -> 推理 -> 评估 -> 可视化。

配置：这是整个工作流的起点。您需要配置整个评估过程，选择要评估的模型和数据集。此外，还可以选择评估策略、计算后端等，并定义显示结果的方式。

推理与评估：在这个阶段，OpenCompass 将会开始对模型和数据集进行并行推理和评估。推理阶段主要是让模型从数据集产生输出，而评估阶段则是衡量这些输出与标准答案的匹配程度。这两个过程会被拆分为多个同时运行的“任务”以提高效率，但请注意，如果计算资源有限，这种策略可能会使评测变得更慢。如果需要了解该问题及解决方案，可以参考 FAQ: 效率。

可视化：评估完成后，OpenCompass 将结果整理成易读的表格，并将其保存为 CSV 和 TXT 文件。你也可以激活飞书状态上报功能，此后可以在飞书客户端中及时获得评测状态报告。 接下来，我们将展示 OpenCompass 的基础用法，展示书生浦语在 C-Eval 基准任务上的评估。它们的配置文件可以在 configs/eval_demo.py 中找到。

## 使用 OpenCompass 评测 internlm2-chat-1_8b 模型在 C-Eval 数据集上的性能

### 环境配置

#### 安装

```code
git clone https://github.com/open-compass/opencompass.git
cd opencompass
pip install -e .
```
#### 数据准备

解压评测数据集到 data/ 处

```code
cp /share/temp/datasets/OpenCompassData-core-20231110.zip /root/opencompass/
unzip OpenCompassData-core-20231110.zip
```
将会在 OpenCompass 下看到data文件夹

### 查看支持的数据集和模型

列出所有跟 InternLM 及 C-Eval 相关的配置

```code
python tools/list_configs.py internlm ceval
```

<img src="config.png" alt="Resized Image 1" width="800"/>

将会看到

```code
+----------------------------------------+----------------------------------------------------------------------+
| Model                                  | Config Path                                                          |
|----------------------------------------+----------------------------------------------------------------------|
| hf_internlm2_1_8b                      | configs/models/hf_internlm/hf_internlm2_1_8b.py                      |
| hf_internlm2_20b                       | configs/models/hf_internlm/hf_internlm2_20b.py                       |
| hf_internlm2_5_1_8b_chat               | configs/models/hf_internlm/hf_internlm2_5_1_8b_chat.py               |
| hf_internlm2_5_20b_chat                | configs/models/hf_internlm/hf_internlm2_5_20b_chat.py                |
| hf_internlm2_5_7b                      | configs/models/hf_internlm/hf_internlm2_5_7b.py                      |
| hf_internlm2_5_7b_chat                 | configs/models/hf_internlm/hf_internlm2_5_7b_chat.py                 |
| hf_internlm2_7b                        | configs/models/hf_internlm/hf_internlm2_7b.py                        |
| hf_internlm2_base_20b                  | configs/models/hf_internlm/hf_internlm2_base_20b.py                  |
| hf_internlm2_base_7b                   | configs/models/hf_internlm/hf_internlm2_base_7b.py                   |
| hf_internlm2_chat_1_8b                 | configs/models/hf_internlm/hf_internlm2_chat_1_8b.py                 |
| hf_internlm2_chat_1_8b_sft             | configs/models/hf_internlm/hf_internlm2_chat_1_8b_sft.py             |
| hf_internlm2_chat_20b                  | configs/models/hf_internlm/hf_internlm2_chat_20b.py                  |
| hf_internlm2_chat_20b_sft              | configs/models/hf_internlm/hf_internlm2_chat_20b_sft.py              |
| hf_internlm2_chat_20b_with_system      | configs/models/hf_internlm/hf_internlm2_chat_20b_with_system.py      |
| hf_internlm2_chat_7b                   | configs/models/hf_internlm/hf_internlm2_chat_7b.py                   |
| hf_internlm2_chat_7b_sft               | configs/models/hf_internlm/hf_internlm2_chat_7b_sft.py               |
| hf_internlm2_chat_7b_with_system       | configs/models/hf_internlm/hf_internlm2_chat_7b_with_system.py       |
| hf_internlm2_chat_math_20b             | configs/models/hf_internlm/hf_internlm2_chat_math_20b.py             |
| hf_internlm2_chat_math_20b_with_system | configs/models/hf_internlm/hf_internlm2_chat_math_20b_with_system.py |
| hf_internlm2_chat_math_7b              | configs/models/hf_internlm/hf_internlm2_chat_math_7b.py              |
| hf_internlm2_chat_math_7b_with_system  | configs/models/hf_internlm/hf_internlm2_chat_math_7b_with_system.py  |
| hf_internlm2_math_20b                  | configs/models/hf_internlm/hf_internlm2_math_20b.py                  |
| hf_internlm2_math_7b                   | configs/models/hf_internlm/hf_internlm2_math_7b.py                   |
| hf_internlm_20b                        | configs/models/hf_internlm/hf_internlm_20b.py                        |
| hf_internlm_7b                         | configs/models/hf_internlm/hf_internlm_7b.py                         |
| hf_internlm_chat_20b                   | configs/models/hf_internlm/hf_internlm_chat_20b.py                   |
| hf_internlm_chat_7b                    | configs/models/hf_internlm/hf_internlm_chat_7b.py                    |
| internlm_7b                            | configs/models/internlm/internlm_7b.py                               |
| lmdeploy_internlm2_1_8b                | configs/models/hf_internlm/lmdeploy_internlm2_1_8b.py                |
| lmdeploy_internlm2_20b                 | configs/models/hf_internlm/lmdeploy_internlm2_20b.py                 |
| lmdeploy_internlm2_5_1_8b_chat         | configs/models/hf_internlm/lmdeploy_internlm2_5_1_8b_chat.py         |
| lmdeploy_internlm2_5_20b_chat          | configs/models/hf_internlm/lmdeploy_internlm2_5_20b_chat.py          |
| lmdeploy_internlm2_5_7b                | configs/models/hf_internlm/lmdeploy_internlm2_5_7b.py                |
| lmdeploy_internlm2_5_7b_chat           | configs/models/hf_internlm/lmdeploy_internlm2_5_7b_chat.py           |
| lmdeploy_internlm2_5_7b_chat_1m        | configs/models/hf_internlm/lmdeploy_internlm2_5_7b_chat_1m.py        |
| lmdeploy_internlm2_7b                  | configs/models/hf_internlm/lmdeploy_internlm2_7b.py                  |
| lmdeploy_internlm2_base_20b            | configs/models/hf_internlm/lmdeploy_internlm2_base_20b.py            |
| lmdeploy_internlm2_base_7b             | configs/models/hf_internlm/lmdeploy_internlm2_base_7b.py             |
| lmdeploy_internlm2_chat_1_8b           | configs/models/hf_internlm/lmdeploy_internlm2_chat_1_8b.py           |
| lmdeploy_internlm2_chat_1_8b_sft       | configs/models/hf_internlm/lmdeploy_internlm2_chat_1_8b_sft.py       |
| lmdeploy_internlm2_chat_20b            | configs/models/hf_internlm/lmdeploy_internlm2_chat_20b.py            |
| lmdeploy_internlm2_chat_20b_sft        | configs/models/hf_internlm/lmdeploy_internlm2_chat_20b_sft.py        |
| lmdeploy_internlm2_chat_7b             | configs/models/hf_internlm/lmdeploy_internlm2_chat_7b.py             |
| lmdeploy_internlm2_chat_7b_sft         | configs/models/hf_internlm/lmdeploy_internlm2_chat_7b_sft.py         |
| lmdeploy_internlm2_series              | configs/models/hf_internlm/lmdeploy_internlm2_series.py              |
| lmdeploy_internlm_20b                  | configs/models/hf_internlm/lmdeploy_internlm_20b.py                  |
| lmdeploy_internlm_7b                   | configs/models/hf_internlm/lmdeploy_internlm_7b.py                   |
| lmdeploy_internlm_chat_20b             | configs/models/hf_internlm/lmdeploy_internlm_chat_20b.py             |
| lmdeploy_internlm_chat_7b              | configs/models/hf_internlm/lmdeploy_internlm_chat_7b.py              |
| ms_internlm_chat_7b_8k                 | configs/models/ms_internlm/ms_internlm_chat_7b_8k.py                 |
| vllm_internlm2_chat_1_8b               | configs/models/hf_internlm/vllm_internlm2_chat_1_8b.py               |
| vllm_internlm2_chat_1_8b_sft           | configs/models/hf_internlm/vllm_internlm2_chat_1_8b_sft.py           |
| vllm_internlm2_chat_20b                | configs/models/hf_internlm/vllm_internlm2_chat_20b.py                |
| vllm_internlm2_chat_20b_sft            | configs/models/hf_internlm/vllm_internlm2_chat_20b_sft.py            |
| vllm_internlm2_chat_7b                 | configs/models/hf_internlm/vllm_internlm2_chat_7b.py                 |
| vllm_internlm2_chat_7b_sft             | configs/models/hf_internlm/vllm_internlm2_chat_7b_sft.py             |
| vllm_internlm2_series                  | configs/models/hf_internlm/vllm_internlm2_series.py                  |
+----------------------------------------+----------------------------------------------------------------------+
+--------------------------------+------------------------------------------------------------------+
| Dataset                        | Config Path                                                      |
|--------------------------------+------------------------------------------------------------------|
| ceval_clean_ppl                | configs/datasets/ceval/ceval_clean_ppl.py                        |
| ceval_contamination_ppl_810ec6 | configs/datasets/contamination/ceval_contamination_ppl_810ec6.py |
| ceval_gen                      | configs/datasets/ceval/ceval_gen.py                              |
| ceval_gen_2daf24               | configs/datasets/ceval/ceval_gen_2daf24.py                       |
| ceval_gen_5f30c7               | configs/datasets/ceval/ceval_gen_5f30c7.py                       |
| ceval_internal_ppl_1cd8bf      | configs/datasets/ceval/ceval_internal_ppl_1cd8bf.py              |
| ceval_internal_ppl_93e5ce      | configs/datasets/ceval/ceval_internal_ppl_93e5ce.py              |
| ceval_ppl                      | configs/datasets/ceval/ceval_ppl.py                              |
| ceval_ppl_1cd8bf               | configs/datasets/ceval/ceval_ppl_1cd8bf.py                       |
| ceval_ppl_578f8d               | configs/datasets/ceval/ceval_ppl_578f8d.py                       |
| ceval_ppl_93e5ce               | configs/datasets/ceval/ceval_ppl_93e5ce.py                       |
| ceval_zero_shot_gen_bd40ef     | configs/datasets/ceval/ceval_zero_shot_gen_bd40ef.py             |
+--------------------------------+------------------------------------------------------------------+
```

### 启动评测 (30% A100 资源)

确保按照上述步骤正确安装 OpenCompass 并准备好数据集后，可以通过以下命令评测 InternLM2-Chat-1.8B 模型在 C-Eval 数据集上的性能。由于 OpenCompass 默认并行启动评估过程，我们可以在第一次运行时以 --debug 模式启动评估，并检查是否存在问题。在 --debug 模式下，任务将按顺序执行，并实时打印输出。

```code
python run.py --datasets ceval_gen --hf-path /share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b --tokenizer-path /share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b --tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True --model-kwargs trust_remote_code=True device_map='auto' --max-seq-len 1024 --max-out-len 16 --batch-size 2 --hf-num-gpus 1 --debug
```

碰到错误：

<img src="error.png" alt="Resized Image 1" width="800"/>

解决方案：

```code
pip install protobuf

export MKL_SERVICE_FORCE_INTEL=1
#或
export MKL_THREADING_LAYER=GNU
```

命令解析

```code
python run.py
--datasets ceval_gen \
--hf-path /share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b \  # HuggingFace 模型路径
--tokenizer-path /share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b \  # HuggingFace tokenizer 路径（如果与模型路径相同，可以省略）
--tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True \  # 构建 tokenizer 的参数
--model-kwargs device_map='auto' trust_remote_code=True \  # 构建模型的参数
--max-seq-len 1024 \  # 模型可以接受的最大序列长度
--max-out-len 16 \  # 生成的最大 token 数
--batch-size 2  \  # 批量大小
--hf-num-gpus 1  # 运行模型所需的 GPU 数量
--debug
```

如果一切正常，您应该看到屏幕上显示 “Starting inference process”：

我们进入log查看

<img src="start.png" alt="Resized Image 1" width="800"/>

评测中日志

<img src="eval.png" alt="Resized Image 1" width="800"/>

评测完成后，将会看到：

```code
dataset                                         version    metric         mode      internlm2-chat-1_8b_hf
----------------------------------------------  ---------  -------------  ------  ------------------------
ceval-computer_network                          db9ce2     accuracy       gen                        36.84
ceval-operating_system                          1c2571     accuracy       gen                        42.11
ceval-computer_architecture                     a74dad     accuracy       gen                        19.05
ceval-college_programming                       4ca32a     accuracy       gen                        35.14
ceval-college_physics                           963fa8     accuracy       gen                        31.58
ceval-college_chemistry                         e78857     accuracy       gen                        37.50
ceval-advanced_mathematics                      ce03e2     accuracy       gen                        26.32
ceval-probability_and_statistics                65e812     accuracy       gen                        44.44
ceval-discrete_mathematics                      e894ae     accuracy       gen                        37.50
ceval-electrical_engineer                       ae42b9     accuracy       gen                        32.43
ceval-metrology_engineer                        ee34ea     accuracy       gen                        62.50
ceval-high_school_mathematics                   1dc5bf     accuracy       gen                        16.67
ceval-high_school_physics                       adf25f     accuracy       gen                        36.84
ceval-high_school_chemistry                     2ed27f     accuracy       gen                        57.89
ceval-high_school_biology                       8e2b9a     accuracy       gen                        26.32
ceval-middle_school_mathematics                 bee8d5     accuracy       gen                        26.32
ceval-middle_school_biology                     86817c     accuracy       gen                        76.19
ceval-middle_school_physics                     8accf6     accuracy       gen                        57.89
ceval-middle_school_chemistry                   167a15     accuracy       gen                        80.00
ceval-veterinary_medicine                       b4e08d     accuracy       gen                        60.87
ceval-college_economics                         f3f4e6     accuracy       gen                        40.00
ceval-business_administration                   c1614e     accuracy       gen                        30.30
ceval-marxism                                   cf874c     accuracy       gen                        73.68
ceval-mao_zedong_thought                        51c7a4     accuracy       gen                        66.67
ceval-education_science                         591fee     accuracy       gen                        55.17
ceval-teacher_qualification                     4e4ced     accuracy       gen                        61.36
ceval-high_school_politics                      5c0de2     accuracy       gen                        52.63
ceval-high_school_geography                     865461     accuracy       gen                        42.11
ceval-middle_school_politics                    5be3e7     accuracy       gen                        80.95
ceval-middle_school_geography                   8a63be     accuracy       gen                        83.33
ceval-modern_chinese_history                    fc01af     accuracy       gen                        56.52
ceval-ideological_and_moral_cultivation         a2aa4a     accuracy       gen                        73.68
ceval-logic                                     f5b022     accuracy       gen                        50.00
ceval-law                                       a110a1     accuracy       gen                        29.17
ceval-chinese_language_and_literature           0f8b68     accuracy       gen                        39.13
ceval-art_studies                               2a1300     accuracy       gen                        51.52
ceval-professional_tour_guide                   4e673e     accuracy       gen                        62.07
ceval-legal_professional                        ce8787     accuracy       gen                        56.52
ceval-high_school_chinese                       315705     accuracy       gen                        42.11
ceval-high_school_history                       7eb30a     accuracy       gen                        65.00
ceval-middle_school_history                     48ab4a     accuracy       gen                        86.36
ceval-civil_servant                             87d061     accuracy       gen                        44.68
ceval-sports_science                            70f27b     accuracy       gen                        47.37
ceval-plant_protection                          8941f9     accuracy       gen                        54.55
ceval-basic_medicine                            c409d6     accuracy       gen                        73.68
ceval-clinical_medicine                         49e82d     accuracy       gen                        45.45
ceval-urban_and_rural_planner                   95b885     accuracy       gen                        41.30
ceval-accountant                                002837     accuracy       gen                        32.65
ceval-fire_engineer                             bc23f5     accuracy       gen                        32.26
ceval-environmental_impact_assessment_engineer  c64e2d     accuracy       gen                        45.16
ceval-tax_accountant                            3a5e3c     accuracy       gen                        38.78
ceval-physician                                 6e277d     accuracy       gen                        36.73
ceval-stem                                      -          naive_average  gen                        42.22
ceval-social-science                            -          naive_average  gen                        58.62
ceval-humanities                                -          naive_average  gen                        55.64
ceval-other                                     -          naive_average  gen                        44.78
ceval-hard                                      -          naive_average  gen                        36.09
ceval                                           -          naive_average  gen                        48.76
```

跟tutorial里略有些出入。

## 使用OpenCompass进行主观评测

类似于客观评测的方式，导入需要评测的datasets。我们在configs文件夹下创建一个eval_subjective_custom.py。

```python
from mmengine.config import read_base

with read_base():
    from .datasets.subjective.alignbench.alignbench_judgeby_critiquellm import alignbench_datasets
```

导入模型模版，分片，推理，评估以及总结参数的指定工具。

```python
from opencompass.models import HuggingFacewithChatTemplate
from opencompass.tasks import OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask
from opencompass.summarizers import SubjectiveSummarizer
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
```

设置我们关注的模型。

```code
models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='internlm2-chat-1_8b',
        path='/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b',
        tokenizer_path='/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b',
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True,
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation='left',
            trust_remote_code=True
        ),
        generation_kwargs=dict(
            do_sample=True,
        ),
        max_out_len=16,
        max_seq_len=1024,
        batch_size=16,
        run_cfg=dict(num_gpus=1)
    )
]
```

因为运行时间原因，我们调大了batch size。由于主观评测的模型设置参数通常与客观评测不同，往往需要设置do_sample的方式进行推理而不是greedy，故可以在配置文件中自行修改相关参数。

指定数据集

```python
datasets = [*alignbench_datasets]
```

judgemodel通常被设置为GPT4等强力模型，可以直接按照config文件中的配置填入自己的API key，或使用自定义的模型作为judgemodel。这里我们选择了一个参数相对没那么多的模型作为评估模型，internlm2-chat-7b。

```python
judge_models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='internlm2-chat-7b',
        path='/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-7b',
        tokenizer_path='/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-7b',
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True,
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation='left',
            trust_remote_code=True
        ),
        generation_kwargs=dict(
            do_sample=True,
        ),
        max_out_len=16,
        max_seq_len=1024,
        batch_size=16,
        run_cfg=dict(num_gpus=1)
    )
]
```

定义主观评估任务的执行方式

```python
eval = dict(
    partitioner=dict(type=SubjectiveNaivePartitioner, models=models, judge_models=judge_models,),
    runner=dict(type=LocalRunner, max_num_workers=128, task=dict(type=SubjectiveEvalTask)),
)
```

定义一个总结器，用于汇总和分析主观评估任务的结果。

```python
summarizer = dict(type=SubjectiveSummarizer, function='subjective')
```

指定工作路径，用于存储评估文件。

```python
work_dir = 'outputs/subjective/'
```

启动评测并输出评测结果

```code
python run.py configs/eval_subjective_custom.py -r --debug
```

-r 参数支持复用模型推理和评估结果, 第一次运行我就没加了。

<img src="run_sub.png" alt="Resized Image 1" width="800"/>

<img src="start_sub.png" alt="Resized Image 1" width="800"/>



