## Llama Index介绍
RAG是不改变模型的权重，只是给模型引入格外的信息。类比人类编程的过程，相当于你阅读函数文档然后短暂的记住了某个函数的用法。

LlamaIndex 是一个上下文增强的 LLM 框架，旨在通过将其与特定上下文数据集集成，增强大型语言模型（LLMs）的能力。它允许您构建应用程序，既利用 LLMs 的优势，又融入您的私有或领域特定信息。具有以下优势：

1. 高效的数据集成：
支持多种数据源，如数据库、文件系统、API 等，使得数据集成过程更加顺畅和无缝。能够处理结构化和非结构化数据，提供灵活的数据输入方式。

2. 灵活的索引结构：
提供多种索引类型，如倒排索引、向量索引等，用户可以根据具体应用场景选择最合适的索引结构。支持自定义索引策略，以满足特定需求。
3. 查询性能优化：
内置查询重写、缓存和并行处理等优化技术，大大提高了查询速度和效率。提供了智能查询路由和负载均衡，确保系统在高并发情况下的稳定性和性能。

4. 可扩展性：
设计考虑了横向扩展，允许在分布式环境中部署，处理大规模数据时依然能够保持高性能。支持集群和分片，能够适应数据量和查询量的增长。

5. 多语言支持：
支持多种自然语言处理任务，如文本分类、信息检索、问答系统等。能够处理多种语言的数据，适用于全球化应用。