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

# 设置嵌入模型选项和模型名称
EMBEDDING_MODEL = "E:/LLM/ModelsUse/paraphrase-multilingual-MiniLM-L12-v2"
# EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
VECTOR_STORE_PATH = "./data/product_vectors"  # 向量存储保存目录名
BATCH_SIZE = 1000  # 批处理大小
SAVE_VECTORS = True  # 是否保存向量存储

class ProductKnowledgeBase:
    def __init__(self, json_file_path: str, max_products: int = None):
        self.products = []
        self.load_data(json_file_path, max_products)
        self.vectorstore = None

    def load_data(self, json_file_path: str, max_products: int = None):
        """从JSONL文件加载产品数据，可选限制产品数量"""
        try:
            print(f"正在加载商品数据文件: {json_file_path}")

            # 检查文件是标准JSON还是JSONL格式
            with open(json_file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                # 重置文件指针
                f.seek(0)

                # 尝试作为标准JSON解析
                try:
                    if first_line.startswith('[') or first_line.startswith('{'):
                        data = json.load(f)

                        # 如果数据是一个列表，直接使用
                        if isinstance(data, list):
                            # 可选限制加载的产品数量
                            if max_products is not None:
                                data = data[:max_products]

                            for item in data:
                                self.products.append(Product(item))
                        # 如果数据是单个产品的字典，放入列表中
                        elif isinstance(data, dict):
                            self.products.append(Product(data))

                        print(f"已加载 {len(self.products)} 个产品（标准JSON格式）。")
                        return
                except json.JSONDecodeError:
                    # 如果不是标准JSON，则按JSONL处理
                    pass

            # 按JSONL格式处理（每行一个JSON对象）
            count = 0
            with open(json_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:  # 跳过空行
                        continue

                    try:
                        # 使用eval解析每行的JSON对象
                        item = eval(line)

                        # 处理商品数据
                        product = Product(item)
                        self.products.append(product)

                        count += 1
                        if count % 10000 == 0:
                            print(f"已加载 {count} 个产品...")

                        # 如果达到限制，停止加载
                        if max_products is not None and count >= max_products:
                            break

                    except Exception as e:
                        print(f"警告: 跳过无效的JSON行: {line[:100]}..., 错误: {str(e)}")

            print(f"已加载 {len(self.products)} 个产品（JSONL格式）。")

        except Exception as e:
            print(f"加载数据文件出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def create_vector_store(self):
        """创建向量存储，支持批处理、进度条和向量存储保存/加载"""
        # FAISS在保存时会创建一个名为VECTOR_STORE_PATH的目录，并在其中保存index.faiss和index.pkl文件
        index_faiss_path = os.path.join(VECTOR_STORE_PATH, "index.faiss")
        index_pkl_path = os.path.join(VECTOR_STORE_PATH, "index.pkl")

        print(f"检查向量存储文件是否存在:")
        print(f"- 目录: {os.path.abspath(VECTOR_STORE_PATH)}")
        print(f"- FAISS索引: {os.path.abspath(index_faiss_path)}")
        print(f"- PKL索引: {os.path.abspath(index_pkl_path)}")

        # 检查是否存在保存的向量存储
        if os.path.exists(VECTOR_STORE_PATH) and os.path.exists(index_faiss_path) and os.path.exists(index_pkl_path):
            print(f"找到现有向量存储文件!")
            try:
                print(f"正在加载向量存储...")
                embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
                self.vectorstore = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
                print(f"向量存储加载成功! 包含 {len(self.vectorstore.index_to_docstore_id)} 个产品文档")
                return
            except Exception as e:
                print(f"加载向量存储失败: {str(e)}")
                print("将重新创建向量存储...")
        else:
            print("未找到现有向量存储文件，将创建新的向量存储")
            if not os.path.exists(VECTOR_STORE_PATH):
                print(f"缺少目录: {VECTOR_STORE_PATH}")
            elif not os.path.exists(index_faiss_path):
                print(f"缺少文件: {index_faiss_path}")
            elif not os.path.exists(index_pkl_path):
                print(f"缺少文件: {index_pkl_path}")

        # 转换产品为文档
        print("准备向量化产品数据...")
        documents = [product.to_document() for product in self.products]
        total_docs = len(documents)
        print(f"共有 {total_docs} 个产品文档需要处理")

        try:
            # 初始化嵌入模型
            print(f"初始化嵌入模型: {EMBEDDING_MODEL}")
            embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

            # 批量处理文档
            print(f"开始批量处理文档，批次大小: {BATCH_SIZE}")
            batches = [documents[i:i + BATCH_SIZE] for i in range(0, total_docs, BATCH_SIZE)]

            # 处理第一批，创建初始向量存储
            print(f"处理第1批文档 (1-{min(BATCH_SIZE, total_docs)})")
            self.vectorstore = FAISS.from_documents(batches[0], embeddings)

            # 处理剩余批次
            if len(batches) > 1:
                for i, batch in enumerate(tqdm(batches[1:], desc="处理批次", unit="批")):
                    batch_num = i + 2  # 第一批已经处理过
                    start_idx = (batch_num - 1) * BATCH_SIZE + 1
                    end_idx = min(batch_num * BATCH_SIZE, total_docs)
                    print(f"处理第{batch_num}批文档 ({start_idx}-{end_idx})")

                    # 处理当前批次
                    batch_vectorstore = FAISS.from_documents(batch, embeddings)

                    # 合并到主向量存储
                    self.vectorstore.merge_from(batch_vectorstore)

            # 保存向量存储
            if SAVE_VECTORS:
                print(f"向量存储创建完成，正在保存到目录: {VECTOR_STORE_PATH}")

                try:
                    # 创建目录(如果不存在)
                    if not os.path.exists(VECTOR_STORE_PATH):
                        os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

                    # 保存向量存储
                    self.vectorstore.save_local(VECTOR_STORE_PATH)
                    print(f"向量存储已保存! 文件位置:")
                    print(f"- {os.path.abspath(index_faiss_path)}")
                    print(f"- {os.path.abspath(index_pkl_path)}")
                except Exception as e:
                    print(f"保存向量存储时出错: {str(e)}")
                    import traceback
                    traceback.print_exc()
            else:
                print("已跳过向量存储保存（SAVE_VECTORS=False）")

            print(f"向量存储创建完成，共包含 {len(self.vectorstore.index_to_docstore_id)} 个产品文档")

        except Exception as e:
            print(f"创建向量存储时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def search_products(self, query: str, k: int = 5) -> List[Dict]:
        """搜索与查询相关的产品"""
        if not self.vectorstore:
            raise ValueError("向量存储未初始化，请先调用create_vector_store()")

        # 使用向量存储进行相似度搜索
        results = self.vectorstore.similarity_search_with_score(query, k=k)

        products_info = []
        for doc, score in results:
            # 转换相似度分数 (FAISS返回的是距离，越小越相似)
            # 将其转换为0-1之间的相似度分数
            similarity = 1.0 / (1.0 + float(score))

            products_info.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": similarity
            })

        return products_info

    def get_product_by_id(self, product_id: str) -> Product:
        """通过ID直接获取产品对象"""
        for product in self.products:
            if product.id == product_id:
                return product
        return None