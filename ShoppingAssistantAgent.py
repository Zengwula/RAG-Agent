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
from ProductKnowledgeBase import ProductKnowledgeBase
from ShoppingTools import ShoppingTools



class ShoppingAssistantAgent:
    def __init__(self, kb: ProductKnowledgeBase):
        self.kb = kb
        self.tools_instance = ShoppingTools(kb)  # 创建工具实例
        self.tools = self.tools_instance.get_tools()

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
            model="deepseek-chat",
            temperature=0.7,
            streaming=True,
            openai_api_key=os.environ["OPENAI_API_KEY"],
            openai_api_base=os.environ["OPENAI_API_BASE"]
        )

    def _create_agent(self):
        """创建带有对话历史记忆的ReAct代理 - 更新提示语"""
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

        其中，Final Answer要以如下格式回答：
        序号.商品名称
        价格：
        评分：
        优点：
        缺点：
        商品特点：
        优点，缺点，商品特点这三栏尽量调用评论分析工具，给的详细一点

        工具名称必须是以下之一: {tool_names}

        注意：
        1. 为用户提供5-8个商品，并总结每个商品的评论，提供优缺点分析。
        2. 如果用户询问之前提到过的商品，请记得之前的对话并给予更详细的分析。
        3. 对于类似"第三款"、"第一个"这样的问题，请调用"获取产品详情"工具，直接输入"第三款"作为参数。
        4. 记录所有推荐过的商品名称和ID，当用户提到这些商品时，确保能找到对应的商品详情。
        5. 当用户提到价格限制和商品类型时，优先使用"按价格和关键词搜索"工具，这比单纯按价格筛选更准确。
        6. 当回复用户时，准确使用商品实际名称，不要擅自翻译或修改商品名称。

        客户问题: {input}
        {agent_scratchpad}
        """)

        agent = create_react_agent(self.llm, self.tools, prompt)

        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            memory=self.memory
        )

    def get_recommended_products(self):
        """获取当前对话中推荐的所有产品"""
        return self.tools_instance.recommended_products

    def run(self, query: str) -> Dict:
        """运行代理回答用户查询"""
        try:
            # 识别并处理"第X款"类型的查询
            import re
            ordinal_pattern = re.compile(r'第(\d+)款')
            match = ordinal_pattern.search(query)

            if match:
                # 提取序号
                index = int(match.group(1))
                print(f"检测到用户询问第{index}款商品")

                # 首先检查工具实例中的序号映射
                if index in self.tools_instance.product_order_map:
                    product_id = self.tools_instance.product_order_map[index]
                    print(f"从映射表中找到第{index}款商品ID: {product_id}")
                    details = self.tools_instance.get_product_details(product_id)
                    result = {"output": details}
                    self.memory.save_context({"input": query}, result)
                    return result

                # 如果映射表中没有，则查找聊天历史
                chat_history = self.memory.load_memory_variables({})["chat_history"]

                # 反向遍历历史记录寻找推荐
                products_list = None
                for i in range(len(chat_history) - 1, -1, -1):
                    if isinstance(chat_history[i], AIMessage) and "找到以下相关产品" in chat_history[i].content:
                        products_list = chat_history[i].content
                        break

                if products_list:
                    # 尝试提取第index款商品的名称或ID
                    sections = products_list.split("\n\n")
                    if 0 < index <= len(sections) - 1:  # -1是因为第一部分通常是介绍文本
                        product_section = sections[index]
                        # 提取名称（第一行通常是产品名称）
                        product_name = product_section.split("\n")[0]
                        if product_name.startswith(f"{index}. "):
                            product_name = product_name[len(f"{index}. "):]

                        # 查找产品ID
                        id_line = None
                        for line in product_section.split("\n"):
                            if "产品ID:" in line:
                                id_line = line.strip()
                                break

                        if id_line:
                            product_id = id_line.split("产品ID:")[1].strip()
                            print(f"从历史记录找到第{index}款商品ID: {product_id}")
                            # 保存到映射表中以便后续使用
                            self.tools_instance.product_order_map[index] = product_id
                            # 直接获取产品详情
                            details = self.tools_instance.get_product_details(product_id)
                            result = {"output": details}
                            self.memory.save_context({"input": query}, result)
                            return result
                        elif product_name:
                            print(f"从历史记录找到第{index}款商品名称: {product_name}")
                            # 使用名称获取详情
                            details = self.tools_instance.get_product_details(product_name)
                            result = {"output": details}
                            self.memory.save_context({"input": query}, result)
                            return result

                # 如果仍然找不到，直接将"第N款"传递给获取产品详情工具
                print(f"直接将'第{index}款'传递给产品详情工具")
                details = self.tools_instance.get_product_details(f"第{index}款")
                result = {"output": details}
                self.memory.save_context({"input": query}, result)
                return result

            # 如果不是"第X款"格式或无法找到对应商品，使用正常代理流程
            return self.agent_executor.invoke({"input": query})

        except Exception as e:
            error_msg = str(e)
            print(f"Agent执行出错: {error_msg}")

            # 提供更有帮助的错误信息
            result = {
                "output": "抱歉，处理您的请求时出现了问题。您是想了解某个特定商品的详情吗？请提供商品的名称或告诉我您想了解之前推荐的哪一款商品。"}
            self.memory.save_context({"input": query}, result)
            return result