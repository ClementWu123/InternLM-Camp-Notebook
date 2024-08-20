# 1 配置LMDeploy环境

## 1.1 InternStudio开发机创建与环境搭建

打开[InternStudio平台](https://studio.intern-ai.org.cn/console/instance)，进入如下界面并按箭头指示顺序点击***创建开发机***。

![img](https://raw.githubusercontent.com/BigWhiteFox/pictures/main/2.PNG)

点选开发机，自拟一个开发机名称，选择***Cuda12.2-conda***镜像。

我们要运行参数量为7B的InternLM2.5，由[InternLM2.5的码仓](https://huggingface.co/internlm/internlm2_5-7b-chat/blob/main/config.json)查询InternLM2.5-7b-chat的config.json文件可知，<a id="权重查询">该模型的权重被存储为`bfloat16`格式</a>

![img](https://raw.githubusercontent.com/BigWhiteFox/pictures/main/3.png)

对于一个7B（70亿）参数的模型，每个参数使用16位浮点数（等于 2个 Byte）表示，则模型的权重大小约为：

**70×10^9 parameters×2 Bytes/parameter=14GB**

**70亿个参数×每个参数占用2个字节=14GB**

所以我们需要大于14GB的显存，选择 ***30%A100\*1***(24GB显存容量)，后选择***立即创建***，等状态栏变成运行中，点击***进入开发机***，我们即可开始部署。

在终端中，让我们输入以下指令，来创建一个名为lmdeploy的conda环境，python版本为3.10，创建成功后激活环境并安装0.5.3版本的lmdeploy及相关包。

```Plain
conda create -n lmdeploy  python=3.10 -y
conda activate lmdeploy
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install timm==1.0.8 openai==1.40.3 lmdeploy[all]==0.5.3
```

## 1.2 InternStudio环境获取模型

为方便文件管理，我们需要一个存放模型的目录，本教程统一放置在`/root/models/`目录。

运行以下命令，创建文件夹并设置开发机共享目录的软链接。

```Plain
mkdir /root/models
ln -s /root/share/new_models//Shanghai_AI_Laboratory/internlm2_5-7b-chat /root/models
ln -s /root/share/new_models/OpenGVLab/InternVL2-26B /root/models
```

此时，我们可以看到`/root/models`中会出现`internlm2_5-7b-chat`和`InternVL2-26B`文件夹。
