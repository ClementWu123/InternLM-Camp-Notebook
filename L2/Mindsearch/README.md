# MindSearch CPU-only 版在github codespace部署

和[原有的CPU版本](https://github.com/InternLM/Tutorial/blob/camp3/docs/L2/MindSearch/readme.md)相比区别是把internstudio换成了github codespace。

随着硅基流动提供了免费的 InternLM2.5-7B-Chat 服务（免费的 InternLM2.5-7B-Chat 真的很香），MindSearch 的部署与使用也就迎来了纯 CPU 版本，进一步降低了部署门槛。那就让我们来一起看看如何使用硅基流动的 API 来部署 MindSearch 吧。

## 1. 创建开发机 & 环境配置

打开[codespace主页](https://github.com/codespaces)，选择blank template。

![image](https://github.com/user-attachments/assets/27e7a7ef-37e0-4614-89bf-c96044e3c6f3)

浏览器会自动在新的页面打开一个web版的vscode。

<img width="1591" alt="image" src="https://github.com/user-attachments/assets/58727fec-8d83-417d-88e5-eedc631444f2">

接下来的操作就和我们使用vscode基本没差别了。

然后我们新建一个目录用于存放 MindSearch 的相关代码，并把 MindSearch 仓库 clone 下来。在终端中运行下面的命令：

```bash
mkdir -p /workspaces/mindsearch
cd /workspaces/mindsearch
git clone https://github.com/InternLM/MindSearch.git
cd MindSearch && git checkout b832275 && cd ..
```

接下来，我们创建一个 conda 环境来安装相关依赖。

```bash
# 创建环境
conda create -n mindsearch python=3.10 -y
# 激活环境
conda activate mindsearch
# 安装依赖
pip install -r /workspaces/mindsearch/MindSearch/requirements.txt

## 2. 获取硅基流动 API Key

因为要使用硅基流动的 API Key，所以接下来便是注册并获取 API Key 了。

首先，我们打开 https://account.siliconflow.cn/login 来注册硅基流动的账号（如果注册过，则直接登录即可）。

在完成注册后，打开 https://cloud.siliconflow.cn/account/ak 来准备 API Key。首先创建新 API 密钥，然后点击密钥进行复制，以备后续使用。

![image](https://github.com/user-attachments/assets/7905a2fc-ef30-4e33-b214-274bebdc9251)

## 3. 启动 MindSearch

### 3.1 启动后端

由于硅基流动 API 的相关配置已经集成在了 MindSearch 中，所以我们可以直接执行下面的代码来启动 MindSearch 的后端。

```bash
export SILICON_API_KEY=第二步中复制的密钥
conda activate mindsearch
cd /workspaces/mindsearch/MindSearch
python -m mindsearch.app --lang cn --model_format internlm_silicon --search_engine DuckDuckGoSearch
```

### 3.2 启动前端

在后端启动完成后，我们打开新终端运行如下命令来启动 MindSearch 的前端。

```bash
conda activate mindsearch
cd /workspaces/mindsearch/MindSearch
python frontend/mindsearch_gradio.py
```

前后端都启动后，我们应该可以看到github自动为这两个进程做端口转发。

<img width="1183" alt="image" src="https://github.com/user-attachments/assets/4ee76ca2-06a5-4145-829a-1310e69c0d83">


由于使用codespace，这里我们不需要使用ssh端口转发了，github会自动提示我们打开一个在公网的前端地址。

<img width="600" alt="image" src="https://github.com/user-attachments/assets/545d5827-6ee3-416a-a913-1be09866f29e">


然后就可以即刻体验啦。


<img width="1489" alt="image" src="https://github.com/user-attachments/assets/28f5658c-19a6-4a46-9bc9-51f4923a012c">

如果遇到了 timeout 的问题，可以按照 [文档](./readme_gpu.md#2-使用-bing-的接口) 换用 Bing 的搜索接口。
