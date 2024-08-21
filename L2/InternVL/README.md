## 写在前面（什么是InternVL）
InternVL 是一种用于多模态任务的深度学习模型，旨在处理和理解多种类型的数据输入，如图像和文本。它结合了视觉和语言模型，能够执行复杂的跨模态任务，比如图文匹配、图像描述生成等。通过整合视觉特征和语言信息，InternVL 可以在多模态领域取得更好的表现

## InternVL 模型总览

![image](https://github.com/user-attachments/assets/d3a013a3-4b0a-4434-99a7-002a797bed09)

对于InternVL这个模型来说，它vision模块就是一个微调过的ViT，llm模块是一个InternLM的模型。对于视觉模块来说，它的特殊之处在Dynamic High Resolution。

## Dynamic High Resolution

InternVL独特的预处理模块：动态高分辨率，是为了让ViT模型能够尽可能获取到更细节的图像信息，提高视觉特征的表达能力。对于输入的图片，首先resize成448的倍数，然后按照预定义的尺寸比例从图片上crop对应的区域。细节如图所示。

![image](https://github.com/user-attachments/assets/c49fef28-0818-432f-bc52-1170e2207f44)

## Pixel Shuffle
Pixel Shuffle在超分任务中是一个常见的操作，PyTorch中有官方实现，即nn.PixelShuffle(upscale_factor) 该类的作用就是将一个tensor中的元素值进行重排列，假设tensor维度为[B, C, H, W], PixelShuffle操作不仅可以改变tensor的通道数，也会改变特征图的大小。

## InternVL 部署微调实践

> 我们选定的任务是让InternVL-2B生成文生图提示词，这个任务需要VLM对图片有格式化的描述并输出。

让我们来一起完成一个用VLM模型进行冷笑话生成，让你的模型说出很逗的冷笑话吧。在这里，我们微调InterenVL使用xtuner。部署InternVL使用lmdeploy。

### 准备InternVL模型
我们使用InternVL2-2B模型。该模型已在share文件夹下挂载好，现在让我们把移动出来。
```bash
cd /root
mkdir -p model

# cp 模型

cp -r /root/share/new_models/OpenGVLab/InternVL2-2B /root/model/
```

### 准备环境

这里我们来手动配置下xtuner。

- 配置虚拟环境

```bash
conda create --name xtuner python=3.10 -y

# 激活虚拟环境（注意：后续的所有操作都需要在这个虚拟环境中进行）
conda activate xtuner

# 安装一些必要的库
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y
# 安装其他依赖
apt install libaio-dev
pip install transformers==4.39.3
pip install streamlit==1.36.0

```

- 安装xtuner

```bash
# 创建一个目录，用来存放源代码
mkdir -p /root/InternLM/code

cd /root/InternLM/code

git clone -b v0.1.23  https://github.com/InternLM/XTuner
```

进入XTuner目录

```bash
cd /root/InternLM/code/XTuner
pip install -e '.[deepspeed]'
```

- 安装LMDeploy
  
```bash
pip install lmdeploy==0.5.3
```

- 安装验证

```bash
xtuner version

##命令

xtuner help
```

![image](https://github.com/user-attachments/assets/e1659d01-ad99-44a6-ae6f-b44dce81cf3c)

### 准备微调数据集

我们这里使用huggingface上的zhongshsh/CLoT-Oogiri-GO据集，特别鸣谢～。

```text
@misc{zhong2023clot,
  title={Let's Think Outside the Box: Exploring Leap-of-Thought in Large Language Models with Creative Humor Generation},
  author={Zhong, Shanshan and Huang, Zhongzhan and Gao, Shanghua and Wen, Weushao and Lin, Liang and Zitnik, Marinka and Zhou, Pan},
  journal={arXiv preprint arXiv:2312.02439},
  year={2023}
}
```

![image](https://github.com/user-attachments/assets/3edb08f7-c0ff-481e-b421-f405101ed29c)

数据集我们从官网下载下来并进行去重，只保留中文数据等操作。并制作成XTuner需要的形式。并已在share里，我们一起从share里挪出数据集。


```bash
## 首先让我们安装一下需要的包
pip install datasets matplotlib Pillow timm

## 让我们把数据集挪出来
cp -r /root/share/new_models/datasets/CLoT_cn_2000 /root/InternLM/datasets/
```

让我们打开数据集的一张图看看，我们选择jsonl里的第一条数据对应的图片。首先我们先把这张图片挪动到InternLM文件夹下面。
```bash
cp InternLM/datasets/CLoT_cn_2000/ex_images/007aPnLRgy1hb39z0im50j30ci0el0wm.jpg InternLM/
```

![image](https://github.com/user-attachments/assets/0c2fec15-3cd3-4eb2-8f98-5b5fb645b0f1)
哈哈，是两只猫在掐架。那我给到的冷笑话回复是什么呢？


