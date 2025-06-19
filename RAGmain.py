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
from Product import Product
from IntelligentShoppingAssistant import IntelligentShoppingAssistant
from ProductKnowledgeBase import ProductKnowledgeBase
from ShoppingTools import ShoppingTools
from ShoppingAssistantAgent import ShoppingAssistantAgent


# 配置DeepSeek API (使用OpenAI兼容接口)
os.environ["OPENAI_API_KEY"] = "Your API_KEY"
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"

# 设置嵌入模型选项和模型名称
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"#自选嵌入模型
# EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
VECTOR_STORE_PATH = "./data/product_vectors"  # 向量存储保存目录名
BATCH_SIZE = 1000  # 批处理大小
SAVE_VECTORS = True  # 是否保存向量存储



if __name__ == "__main__":
    import os

    # 替换为您的JSON知识库文件路径
    json_file_path = "需要用dataprocess文件将其转化为特定格式的json文件"

    # max_products = 5000  # 仅加载前5000个产品进行测试
    max_products = None  # 加载所有产品

    # 检查向量存储的保存/加载是否工作
    index_faiss_path = os.path.join(VECTOR_STORE_PATH, "index.faiss")
    index_pkl_path = os.path.join(VECTOR_STORE_PATH, "index.pkl")

    vector_files_exist = os.path.exists(VECTOR_STORE_PATH) and os.path.exists(index_faiss_path) and os.path.exists(
        index_pkl_path)
    if vector_files_exist:
        print(f"发现已保存的向量存储文件:")
        print(f"- 目录: {os.path.abspath(VECTOR_STORE_PATH)}")
        print(f"- FAISS索引: {os.path.abspath(index_faiss_path)}")
        print(f"- PKL索引: {os.path.abspath(index_pkl_path)}")
    else:
        print(f"未找到向量存储文件，将在处理后创建新文件")

    # 初始化购物助手
    assistant = IntelligentShoppingAssistant(json_file_path, max_products)

    # 交互式聊天循环
    print("智能导购助手已启动! 输入'退出'结束对话，输入'历史'查看对话历史。")
    while True:
        user_input = input("\n您的问题: ")
        if user_input.lower() in ["退出", "exit", "quit"]:
            print("谢谢使用，再见!")
            break
        elif user_input.lower() in ["历史", "history"]:
            print(assistant.view_history())
            continue

        response = assistant.chat(user_input)
        print(f"\n助手: {response['output']}")