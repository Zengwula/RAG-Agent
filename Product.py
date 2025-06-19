import os
import json
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document, HumanMessage, AIMessage
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel, Field
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI

# 设置嵌入模型选项和模型名称
EMBEDDING_MODEL = "E:/LLM/ModelsUse/paraphrase-multilingual-MiniLM-L12-v2"
# EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
VECTOR_STORE_PATH = "./data/product_vectors"  # 向量存储保存目录名
BATCH_SIZE = 1000  # 批处理大小
SAVE_VECTORS = True  # 是否保存向量存储


class Product:
    def __init__(self, data: Dict[str, Any]):
        self.id = data.get("asin", data.get("id", ""))
        self.title = data.get("title", "")
        self.brand = data.get("brand", "")
        self.price = data.get("price", 0.0)
        self.categories = data.get("categories", "")
        self.description = data.get("description", "")
        self.reviews = data.get("reviews", [])
        self.review_count = data.get("review_count", 0)
        self.avg_rating = data.get("avg_rating", 0.0)
        self.primary_category = data.get("primary_category", "")
        self.sales_rank = data.get("sales_rank", 0)

    def to_document(self) -> Document:
        """将产品转换为Document对象以供向量化"""
        metadata = {
            "id": self.id,
            "title": self.title,
            "brand": self.brand,
            "price": self.price,
            "primary_category": self.primary_category,
            "avg_rating": self.avg_rating,
            "review_count": self.review_count
        }

        # 构建内容文本
        content = f"标题: {self.title}\n"
        content += f"品牌: {self.brand}\n"
        content += f"价格: {self.price}\n"
        content += f"类别: {self.categories}\n"

        if self.description:
            content += f"描述: {self.description}\n"

        # 添加评论信息
        if self.reviews:
            content += "评论:\n"
            for review in self.reviews:
                content += f"- 评分: {review.get('overall', 0)}, 内容: {review.get('reviewText', '')}\n"

        return Document(page_content=content, metadata=metadata)