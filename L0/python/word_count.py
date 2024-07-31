import re
from collections import defaultdict

def wordcount(text):
    # 去掉标点符号，并转换为小写
    text = re.sub(r'[^\w\s]', '', text).lower()
    words = text.split()
    
    # 使用字典来统计单词出现次数
    word_count = defaultdict(int)
    for word in words:
        word_count[word] += 1
    
    return dict(word_count)

# 示例输入
text = """Hello world!  
This is an example.  
Word count is fun.  
Is it fun to count words?  
Yes, it is fun!"""

# 调用函数并打印结果
print(wordcount(text))