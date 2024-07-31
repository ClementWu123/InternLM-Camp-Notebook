## Part 1: Word Count输出

<img src="wordcount1.png" alt="Resized Image 1" width="500"/>
<img src="wordcount2.png" alt="Resized Image 2" width="500"/>

## Part 2: Debug笔记

### 我们在代码中插入断点：

断点位置 6 行：文本处理之前<br />
断点位置 8 行：文本分割成单词之后<br />
断点位置 10 行：单词计数过程中<br />

### 使用断点调试

1. 检查文本处理：

```python
text = re.sub(r'[^\w\s]', '', text).lower()
```

2. 检查单词列表：

```python
words = text.split()
```

3. 检查单词计数：

```python
word_count[word] += 1
```

### 调试笔记和截图

1. 初始代码，我们得到一个名为text的local变量



