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
from ShoppingAssistantAgent import ShoppingAssistantAgent

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

            self.conversation_history.append({"role": "user", "content": user_query})

            # 调用代理执行
            result = self.agent.run(user_query)

            # 确保返回字典格式
            if isinstance(result, dict) and "output" in result:

                self.conversation_history.append({"role": "assistant", "content": result["output"]})
                return result
            elif isinstance(result, str):
                output = {"output": result}

                self.conversation_history.append({"role": "assistant", "content": result})
                return output
            else:
                output = {"output": str(result)}

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