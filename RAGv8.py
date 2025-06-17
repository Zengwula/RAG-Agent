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

# 配置DeepSeek API (使用OpenAI兼容接口)
os.environ["OPENAI_API_KEY"] = "你的deepseek API key"
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"

# 设置嵌入模型选项和模型名称
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  
#EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
VECTOR_STORE_PATH = "./data/product_vectors"  # 向量存储保存目录名
BATCH_SIZE = 1000  # 批处理大小
SAVE_VECTORS = True  # 是否保存向量存储


# 定义产品模型
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


# 1. 数据加载和处理
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

                
                try:
                    if first_line.startswith('[') or first_line.startswith('{'):
                        data = json.load(f)

                      
                        if isinstance(data, list):
                            if max_products is not None:
                                data = data[:max_products]

                            for item in data:
                                self.products.append(Product(item))
                       
                        elif isinstance(data, dict):
                            self.products.append(Product(data))

                        print(f"已加载 {len(self.products)} 个产品（标准JSON格式）。")
                        return
                except json.JSONDecodeError:
                    pass

            count = 0
            with open(json_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:  
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
            # 将其转换为0-1之间的相似度分数
            similarity = 1.0 / (1.0 + float(score))

            products_info.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": similarity
            })

        return products_info


# 2. 定义代理需要的工具
class ShoppingTools:
    def __init__(self, knowledge_base: ProductKnowledgeBase):
        self.kb = knowledge_base

    def search_product_tool(self, query: str) -> str:
        """搜索产品的工具"""
        try:
            results = self.kb.search_products(query, k=8)  # 增加搜索结果数量至8个
            if not results:
                return "未找到相关产品。"

            response = "找到以下相关产品：\n\n"
            for i, result in enumerate(results, 1):
                metadata = result["metadata"]
                relevance = result["relevance_score"]

                response += f"{i}. {metadata['title']}\n"
                response += f"   品牌: {metadata['brand']}\n"
                response += f"   价格: ¥{metadata['price']}\n"
                response += f"   类别: {metadata['primary_category']}\n"
                response += f"   评分: {metadata['avg_rating']} ({metadata['review_count']}条评论)\n"
                response += f"   产品ID: {metadata['id']}\n"
                response += f"   相关度: {relevance:.2f}\n\n"

            return response
        except Exception as e:
            return f"搜索产品时出错: {str(e)}"

    def get_product_details(self, product_id: str) -> str:
        """获取产品详细信息的工具"""
        try:
            # 清理产品ID（移除可能的空格和引号）
            product_id = product_id.strip().strip('"\'')

            # 首先尝试使用向量存储查找产品
            try:
                results = self.kb.search_products(f"ID:{product_id}", k=1)
                if results and results[0]["relevance_score"] > 0.5:
                    # 从向量存储结果中获取产品信息
                    metadata = results[0]["metadata"]

                    response = f"产品详情 - {metadata['title']}\n\n"
                    response += f"ID: {metadata['id']}\n"
                    response += f"品牌: {metadata['brand']}\n"
                    response += f"价格: ¥{metadata['price']}\n"
                    response += f"类别: {metadata.get('primary_category', '未分类')}\n"
                    response += f"评分: {metadata['avg_rating']} ({metadata['review_count']}条评论)\n\n"

                    # 从内容中提取描述和其他详细信息
                    content = results[0]["content"]
                    if "描述:" in content:
                        desc_parts = content.split("描述:")[1].split("\n")[0].strip()
                        response += f"描述: {desc_parts}\n\n"
                    else:
                        response += "描述: 暂无\n\n"

                    # 尝试提取评论
                    if "评论:" in content:
                        response += "用户评论摘要:\n"
                        review_section = content.split("评论:")[1]
                        review_lines = review_section.split("\n")

                        # 显示最多5条评论摘要
                        review_count = 0
                        for line in review_lines:
                            if line.startswith("- 评分:") and review_count < 5:
                                response += f"{line}\n"
                                review_count += 1

                    return response
            except Exception as vector_error:
                print(f"向量搜索错误: {str(vector_error)}")
                # 如果向量搜索失败，继续使用原始方法
                pass

            # 如果向量搜索未找到匹配，则继续尝试从产品列表查找
            print(f"正在从产品列表中查找ID: '{product_id}'")
            print(f"产品列表长度: {len(self.kb.products)}")

            # 从产品列表中查找
            for product in self.kb.products:
                # 同时检查id和asin (如果存在)
                prod_id = getattr(product, 'id', '')
                prod_asin = getattr(product, 'asin', '')

                if prod_id == product_id or prod_asin == product_id:
                    response = f"产品详情 - {product.title}\n\n"
                    response += f"ID: {product.id}\n"
                    response += f"品牌: {product.brand}\n"
                    response += f"价格: ¥{product.price}\n"
                    response += f"类别: {product.categories}\n"

                    if product.description:
                        response += f"描述: {product.description}\n\n"
                    else:
                        response += "描述: 暂无\n\n"

                    response += f"评分: {product.avg_rating} ({product.review_count}条评论)\n"
                    response += f"销售排名: {product.sales_rank}\n\n"

                    # 增强评论分析
                    if product.reviews:
                        response += "用户评论分析:\n"

                        # 按评分分类评论
                        positive_reviews = [r for r in product.reviews if r.get('overall', 0) >= 4]
                        neutral_reviews = [r for r in product.reviews if r.get('overall', 0) == 3]
                        negative_reviews = [r for r in product.reviews if r.get('overall', 0) < 3]

                        # 统计信息
                        response += f"- 好评({len(positive_reviews)}条): "
                        if positive_reviews:
                            response += f"{positive_reviews[0].get('reviewText', '')[:100]}...\n"
                        else:
                            response += "暂无\n"

                        response += f"- 中评({len(neutral_reviews)}条): "
                        if neutral_reviews:
                            response += f"{neutral_reviews[0].get('reviewText', '')[:100]}...\n"
                        else:
                            response += "暂无\n"

                        response += f"- 差评({len(negative_reviews)}条): "
                        if negative_reviews:
                            response += f"{negative_reviews[0].get('reviewText', '')[:100]}...\n"
                        else:
                            response += "暂无\n"

                        # 完整评论列表
                        response += "\n所有评论:\n"
                        for i, review in enumerate(product.reviews[:5], 1):  # 最多显示5条
                            response += f"{i}. 评分: {review.get('overall', 0)}, 日期: {review.get('date', '')}\n"
                            response += f"   内容: {review.get('reviewText', '')}\n\n"

                        if len(product.reviews) > 5:
                            response += f"(共有{len(product.reviews)}条评论，仅显示前5条)\n"

                    return response

            # 如果仍然找不到产品，提供友好的错误消息
            return f"未找到ID为'{product_id}'的产品。请尝试使用搜索功能查找相关产品。"
        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            print(f"获取产品详情时出错: {str(e)}\n{trace}")
            return f"获取产品详情时出错: {str(e)}"

    def filter_products_by_price(self, price_range: str) -> str:
        """按价格范围筛选产品的工具"""
        try:
            # 解析价格范围
            parts = price_range.split(',')
            if len(parts) != 2:
                return "价格范围格式不正确，请使用逗号分隔最低价格和最高价格，例如：'100,200'"

            min_price = float(parts[0].strip())
            max_price = float(parts[1].strip())

            filtered_products = [p for p in self.kb.products if min_price <= p.price <= max_price]

            if not filtered_products:
                return f"未找到价格在 ¥{min_price} - ¥{max_price} 范围内的产品。"

            response = f"价格在 ¥{min_price} - ¥{max_price} 范围内的产品：\n\n"
            for i, product in enumerate(filtered_products[:12], 1):  # 最多显示12个结果
                response += f"{i}. {product.title}\n"
                response += f"   品牌: {product.brand}\n"
                response += f"   价格: ¥{product.price}\n"
                response += f"   ID: {product.id}\n\n"

            if len(filtered_products) > 12:
                response += f"(共找到 {len(filtered_products)} 个结果，只显示前12个)\n"

            return response
        except Exception as e:
            return f"按价格筛选产品时出错: {str(e)}"

    def analyze_product_reviews(self, product_id: str) -> str:
        """分析产品评论的工具"""
        try:
            # 查找指定ID的产品
            for product in self.kb.products:
                if product.id == product_id:
                    if not product.reviews:
                        return f"产品 '{product.title}' 暂无评论。"

                    # 评论统计
                    total = len(product.reviews)
                    ratings = [r.get('overall', 0) for r in product.reviews]
                    avg_rating = sum(ratings) / total if total > 0 else 0

                    # 按评分分组
                    rating_groups = {}
                    for rating in range(1, 6):
                        count = ratings.count(rating)
                        percentage = (count / total) * 100 if total > 0 else 0
                        rating_groups[rating] = {
                            'count': count,
                            'percentage': percentage
                        }

                    # 生成分析报告
                    response = f"产品 '{product.title}' 评论分析\n\n"
                    response += f"总评论数: {total}\n"
                    response += f"平均评分: {avg_rating:.1f}/5.0\n\n"

                    response += "评分分布:\n"
                    for rating in range(5, 0, -1):
                        stars = "★" * rating + "☆" * (5 - rating)
                        count = rating_groups[rating]['count']
                        percentage = rating_groups[rating]['percentage']
                        response += f"{stars} ({rating}分): {count}条 ({percentage:.1f}%)\n"

                    # 提取正面和负面评论
                    positive = [r for r in product.reviews if r.get('overall', 0) >= 4]
                    negative = [r for r in product.reviews if r.get('overall', 0) <= 2]

                    if positive:
                        response += "\n正面评论摘要:\n"
                        for i, review in enumerate(positive[:3], 1):
                            response += f"{i}. \"{review.get('reviewText', '')[:100]}...\"\n"

                    if negative:
                        response += "\n负面评论摘要:\n"
                        for i, review in enumerate(negative[:3], 1):
                            response += f"{i}. \"{review.get('reviewText', '')[:100]}...\"\n"

                    return response

            return f"未找到ID为{product_id}的产品。"
        except Exception as e:
            return f"分析产品评论时出错: {str(e)}"

    def get_tools(self) -> List[Tool]:
        """获取所有工具列表"""
        return [
            Tool(
                name="搜索产品",
                func=self.search_product_tool,
                description="根据用户的描述、需求或关键词搜索相关产品。输入应该是描述产品的文本查询。"
            ),
            Tool(
                name="获取产品详情",
                func=self.get_product_details,
                description="获取特定产品的详细信息和评论分析。输入应该是产品ID。"
            ),
            Tool(
                name="按价格筛选产品",
                func=self.filter_products_by_price,
                description="按价格范围筛选产品。输入应该是两个数字，用逗号分隔，表示最低价格和最高价格。例如：'10,50'表示10元到50元之间的产品。"
            ),
            Tool(
                name="分析产品评论",
                func=self.analyze_product_reviews,
                description="分析特定产品的评论，提取关键观点和情感。输入应该是产品ID。"
            )
        ]


# 3. 创建DeepSeek LLM和Agent，添加对话历史功能
class ShoppingAssistantAgent:
    def __init__(self, kb: ProductKnowledgeBase):
        self.kb = kb
        self.tools = ShoppingTools(kb).get_tools()

        # 初始化对话历史记忆
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )

        self.llm = self._create_llm()
        self.agent_executor = self._create_agent()

    def _create_llm(self):
        """创建DeepSeek LLM (通过OpenAI兼容接口)"""
        return ChatOpenAI(
            model="deepseek-chat",  # 指向 DeepSeek-V3-0324 模型
            temperature=0.7,
            streaming=True,
            openai_api_key=os.environ["OPENAI_API_KEY"],
            openai_api_base=os.environ["OPENAI_API_BASE"]
        )

    def _create_agent(self):
        """创建带有对话历史记忆的ReAct代理"""
        prompt = PromptTemplate.from_template("""
        你是一个专业的智能导购助手，你的任务是帮助客户查找、比较和推荐产品。

        你有以下工具可以使用：
        {tools}

        以下是您与用户的对话历史:
        {chat_history}

        请根据对话历史和当前问题，帮助用户找到合适的商品。

        使用以下格式回答：

        Question: 客户的问题
        Thought: 分析问题并确定需要使用哪个工具
        Action: 工具名称
        Action Input: 工具的输入参数
        Observation: 工具返回的结果
        Thought: 基于观察结果进行分析，看是否需要使用其他工具或直接回答
        ... (Thought/Action/Observation可以重复多次)
        Final Answer: 最终对客户的回复

        工具名称必须是以下之一: {tool_names}

        注意：
        1.为用户提供5-8个商品，并总结每个商品的评论，提供优缺点分析。
        2.如果用户询问之前提到过的商品，请记得之前的对话并给予更详细的分析。
        3.



        客户问题: {input}
        {agent_scratchpad}
        """)

        agent = create_react_agent(self.llm, self.tools, prompt)

        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            memory=self.memory  # 添加记忆
        )

    def run(self, query: str) -> Dict:
        """运行代理回答用户查询"""
        try:
            # 使用带有记忆的Agent执行器处理查询
            return self.agent_executor.invoke({"input": query})
        except Exception as e:
            error_msg = str(e)
            print(f"Agent执行出错: {error_msg}")

            # 如果是解析错误，尝试提取有用的部分
            if "Invalid or incomplete response" in error_msg and "Final Answer:" in error_msg:
                # 提取Final Answer部分
                try:
                    final_answer = error_msg.split("Final Answer:")[1].strip()
                    result = {"output": final_answer}

                    # 即使出错也保存到对话历史
                    self.memory.save_context({"input": query}, result)
                    return result
                except:
                    pass

            # 返回默认响应
            result = {"output": "抱歉，处理您的请求时出现了问题。请尝试重新表述您的问题，或者提供更多细节。"}
            self.memory.save_context({"input": query}, result)
            return result


# 4. 主应用类
class IntelligentShoppingAssistant:
    def __init__(self, json_file_path: str, max_products: int = None):
        # 初始化知识库
        self.kb = ProductKnowledgeBase(json_file_path, max_products)
        self.kb.create_vector_store()

        # 初始化代理
        self.agent = ShoppingAssistantAgent(self.kb)

        # 用于调试的对话历史
        self.conversation_history = []

    def chat(self, user_query: str) -> Dict:
        """与用户聊天的主入口函数"""
        try:
            # 保存用户输入到调试历史
            self.conversation_history.append({"role": "user", "content": user_query})

            # 调用代理执行
            result = self.agent.run(user_query)

            # 确保返回字典格式
            if isinstance(result, dict) and "output" in result:
                # 保存助手回复到调试历史
                self.conversation_history.append({"role": "assistant", "content": result["output"]})
                return result
            elif isinstance(result, str):
                output = {"output": result}
                # 保存助手回复到调试历史
                self.conversation_history.append({"role": "assistant", "content": result})
                return output
            else:
                output = {"output": str(result)}
                # 保存助手回复到调试历史
                self.conversation_history.append({"role": "assistant", "content": str(result)})
                return output

        except Exception as e:
            # 错误处理，确保返回字典
            error_msg = str(e)
            print(f"Agent执行出错: {error_msg}")

            # 提取错误中的有用信息
            if "Invalid or incomplete response" in error_msg and "Thought:" in error_msg:
                # 尝试提取思考内容
                thought = error_msg.split("Thought:")[1].split("\n")[0].strip()
                output = {"output": f"处理您的请求时出现问题。我正在思考：{thought}"}
                self.conversation_history.append({"role": "assistant", "content": output["output"]})
                return output

            output = {"output": "抱歉，处理您的请求时出现了问题。请尝试重新表述您的问题。"}
            self.conversation_history.append({"role": "assistant", "content": output["output"]})
            return output

    def view_history(self) -> str:
        """查看对话历史（调试用）"""
        if not self.conversation_history:
            return "尚无对话历史"

        result = "对话历史:\n\n"
        for message in self.conversation_history:
            role = "用户" if message["role"] == "user" else "助手"
            result += f"[{role}]: {message['content']}\n\n"

        return result


# 使用示例
if __name__ == "__main__":
    import os

    # 替换为您的JSON知识库文件路径
    json_file_path = "E:/agent/AgentShop/new_data/product_knowledge_base.json"

    # 可选参数: 限制加载的产品数量，用于测试
    # 取消注释下面一行并设置合适的数量来加速测试
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
