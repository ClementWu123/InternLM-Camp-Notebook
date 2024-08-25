# 0 茴香豆介绍

<div align="center">

![](https://raw.githubusercontent.com/fzd9752/pic_img/main/imgs/image-1.png)

</div>

[**茴香豆**](https://github.com/InternLM/HuixiangDou/) 是由书生·浦语团队开发的一款开源、专门针对国内企业级使用场景设计并优化的知识问答工具。在基础 RAG 课程中我们了解到，RAG 可以有效的帮助提高 LLM 知识检索的相关性、实时性，同时避免 LLM 训练带来的巨大成本。在实际的生产和生活环境需求，对 RAG 系统的开发、部署和调优的挑战更大，如需要解决群应答、能够无关问题拒答、多渠道应答、更高的安全性挑战。因此，根据大量国内用户的实际需求，总结出了**三阶段Pipeline**的茴香豆知识问答助手架构，帮助企业级用户可以快速上手安装部署。

**茴香豆特点**：

* 三阶段 Pipeline （前处理、拒答、响应），提高相应准确率和安全性

* 打通微信和飞书群聊天，适合国内知识问答场景

* 支持各种硬件配置安装，安装部署限制条件少

* 适配性强，兼容多个 LLM 和 API

* 傻瓜操作，安装和配置方便



<div align="center">

![](<https://raw.githubusercontent.com/fzd9752/pic_img/main/imgs/Screenshot 2024-08-23 at 20.23.25.png>)

</div>


# 茴香豆本地标准版搭建

## &#x20;2.1 环境搭建

### 2.1.1 配置服务器



首先登录 [InternStudio](https://studio.intern-ai.org.cn/console/dashboard) ，选择创建开发机：

<div align="center">

![](<https://raw.githubusercontent.com/fzd9752/pic_img/main/imgs/Screenshot 2024-08-24 at 03.01.37.png>)

</div>

镜像选择 `Cuda11.7-conda` ，资源类型选择 `30% A\*100`。输入开发机名称 `huixiangdou`, 点击立即创建。

<div align="center">

![](<https://raw.githubusercontent.com/fzd9752/pic_img/main/imgs/Screenshot 2024-08-24 at 03.03.45.png>)

</div>

在 `开发机` 页面选择刚刚创建的个人开发机 `huixiangdou`，单击 `启动`：

<div align="center">

![](<https://raw.githubusercontent.com/fzd9752/pic_img/main/imgs/Screenshot 2024-08-24 at 14.08.15.png>)

</div>

等服务器准备好开发机资源后，点击 `进入开发机`，继续进行开发环境的搭建。

### 2.1.2 搭建茴香豆虚拟环境

命令行中输入一下命令，创建茴香豆专用 conda 环境：

```bash
studio-conda -o internlm-base -t huixiangdou
```

创建成功，用下面的命令激活环境：

```bash
conda activate huixiangdou
```
<div align="center">

![](<https://raw.githubusercontent.com/fzd9752/pic_img/main/imgs/Screenshot 2024-08-24 at 15.11.37.png>)

</div>

环境激活成功后，命令行前的括号内会显示正在使用的环境，请确保所有茴香豆操作指令在 `huixiangdou` 环境下运行。

## 2.2  安装茴香豆

下面开始茴香豆本地标准版的安装。

### 2.2.1 下载茴香豆

先从茴香豆仓库拉取代码到服务器：

```bash
cd /root
# 克隆代码仓库
git clone https://github.com/internlm/huixiangdou && cd huixiangdou
git checkout 79fa810
```

拉取完成后进入茴香豆文件夹，开始安装。

<div align="center">

![](<https://raw.githubusercontent.com/fzd9752/pic_img/main/imgs/Screenshot 2024-08-24 at 15.16.53.png>)

</div>

### 2.2.2 安装茴香豆所需依赖

首先安装茴香豆所需依赖：

```bash
conda activate huixiangdou
# parsing `word` format requirements
apt update
apt install python-dev libxml2-dev libxslt1-dev antiword unrtf poppler-utils pstotext tesseract-ocr flac ffmpeg lame libmad0 libsox-fmt-mp3 sox libjpeg-dev swig libpulse-dev
# python requirements
pip install BCEmbedding==0.15 cmake==3.30.2 lit==18.1.8 sentencepiece==0.2.0 protobuf==5.27.3 accelerate==0.33.0
pip install -r requirements.txt
# python3.8 安装 faiss-gpu 而不是 faiss
```

### 2.2.3 下载模型文件

茴香豆默认会根据配置文件自动下载对应的模型文件，为了节省时间，本次教程所需的模型已经提前下载到服务器中，我们只需要为本次教程所需的模型建立软连接，然后在配置文件中设置相应路径就可以：

```bash
# 创建模型文件夹
cd /root && mkdir models

# 复制BCE模型
ln -s /root/share/new_models/maidalun1020/bce-embedding-base_v1 /root/models/bce-embedding-base_v1
ln -s /root/share/new_models/maidalun1020/bce-reranker-base_v1 /root/models/bce-reranker-base_v1

# 复制大模型参数（下面的模型，根据作业进度和任务进行**选择一个**就行）
ln -s /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-7b /root/models/internlm2-chat-7b
```

完成后可以在相应目录下看到所需模型文件。

<div align="center">

![](<https://raw.githubusercontent.com/fzd9752/pic_img/main/imgs/Screenshot 2024-08-24 at 15.41.56.png>)

</div>

### 2.2.4 更改配置文件

茴香豆的所有功能开启和模型切换都可以通过 `config.ini` 文件进行修改，默认参数如下：

<div align="center">

![](<https://raw.githubusercontent.com/fzd9752/pic_img/main/imgs/Screenshot 2024-08-24 at 15.46.47.png>)

</div>

执行下面的命令更改配置文件，让茴香豆使用本地模型：

```bash
sed -i '9s#.*#embedding_model_path = "/root/models/bce-embedding-base_v1"#' /root/huixiangdou/config.ini
sed -i '15s#.*#reranker_model_path = "/root/models/bce-reranker-base_v1"#' /root/huixiangdou/config.ini
sed -i '43s#.*#local_llm_path = "/root/models/internlm2-chat-7b"#' /root/huixiangdou/config.ini
```

也可以用编辑器手动修改，文件位置为 `/root/huixiangdou/config.ini`。

修改后的配置文件如下：

<div align="center">

![](<https://raw.githubusercontent.com/fzd9752/pic_img/main/imgs/Screenshot 2024-08-24 at 15.47.17.png>)

</div>

> 注意！配置文件默认的模型和下载好的模型相同。如果不修改地址为本地模型地址，茴香豆将自动从 huggingface hub 拉取模型。如果选择拉取模型的方式，需要提前在命令行中运行 huggingface-cli login 命令，验证 huggingface 权限。


## 2.3 知识库创建

修改完配置文件后，就可以进行知识库的搭建，本次教程选用的是英伟达的Lidar_AI_Solution的文档，利用茴香豆搭建一个Lidar_AI_Solution的知识问答助手。

```bash
conda activate huixiangdou

cd /root/huixiangdou && mkdir repodir

git clone https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution.git --depth=1 repodir/huixiangdou

# Save the features of repodir to workdir, and update the positive and negative example thresholds into `config.ini`
mkdir workdir
python3 -m huixiangdou.service.feature_store
```

在 huixiangdou 文件加下创建 repodir 文件夹，用来储存知识库原始文档。再创建一个文件夹 workdir 用来存放原始文档特征提取到的向量知识库。

<div align="center">

![](<https://raw.githubusercontent.com/fzd9752/pic_img/main/imgs/Screenshot 2024-08-24 at 16.09.01.png>)

</div>

## Gradio UI 界面测试

茴香豆也用 `gradio` 搭建了一个 Web UI 的测试界面，用来测试本地茴香豆助手的效果。

本节课程中，茴香豆助手搭建在远程服务器上，因此需要先建立本地和服务器之间的透传，透传默认的端口为 `7860`，在本地机器命令行中运行如下命令：

```bash
ssh -CNg -L 7860:127.0.0.1:7860 root@ssh.intern-ai.org.cn -p <你的ssh端口号>
```

在运行茴香豆助手的服务器端，输入下面的命令，启动茴香豆 Web UI：

```bash
conda activate huixiangdou
cd /root/huixiangdou
python3 -m huixiangdou.gradio
```

<div align="center">

![](<https://raw.githubusercontent.com/fzd9752/pic_img/main/imgs/Screenshot 2024-08-24 at 17.08.47.png>)

</div>

看到上图相同的结果，说明 `Gradio` 服务启动成功，在本地浏览器中输入 `127.0.0.1:7860` 打开茴香豆助手测试页面：

<div align="center">

![](<https://raw.githubusercontent.com/fzd9752/pic_img/main/imgs/Screenshot 2024-08-24 at 17.09.58.png>)

</div>

现在就可以用页面测试一下茴香豆的交互效果了。
