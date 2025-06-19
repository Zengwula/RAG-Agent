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


class ShoppingTools:
    def __init__(self, knowledge_base: ProductKnowledgeBase):
        self.kb = knowledge_base
        # 新增：保存推荐过的产品信息
        self.recommended_products = {}  # ID -> 产品信息
        self.recommended_products_by_name = {}  # 名称 -> 产品信息

        # 新增：序号到产品ID的映射
        self.product_order_map = {}  # 序号 -> 产品ID

        # 新增：中文名称和ID的映射
        self.translated_names = {}  # 中文名称 -> 产品ID

        # 新增：保存最后一次搜索结果
        self.last_search_results = []

    def search_product_tool(self, query: str) -> str:
        """搜索产品的工具，同时记录推荐的产品信息"""
        try:
            results = self.kb.search_products(query, k=20)
            if not results:
                return "未找到相关产品。"

            response = "找到以下相关产品：\n\n"

            # 清空之前的推荐结果（如果是新的搜索）
            if not query.startswith("ID:"):
                self.recommended_products = {}
                self.recommended_products_by_name = {}
                self.product_order_map = {}
                self.translated_names = {}
                self.last_search_results = results.copy()  # 保存结果副本

            for i, result in enumerate(results, 1):
                metadata = result["metadata"]
                relevance = result["relevance_score"]
                product_id = metadata['id']
                product_title = metadata['title']

                # 保存推荐产品信息到内存中
                self.recommended_products[product_id] = {
                    "metadata": metadata,
                    "content": result["content"]
                }
                # 同时以标题为键保存，用于模糊匹配
                self.recommended_products_by_name[product_title.lower()] = {
                    "id": product_id,
                    "metadata": metadata,
                    "content": result["content"]
                }

                # 保存序号到产品ID的映射
                self.product_order_map[i] = product_id

                # 生成可能的中文名称并保存映射
                chinese_name = self._generate_chinese_name(product_title, metadata['price'])
                if chinese_name:
                    self.translated_names[chinese_name] = product_id
                    # 为第N款格式生成映射
                    self.translated_names[f"第{i}款"] = product_id

                response += f"{i}. {product_title}\n"
                response += f"   品牌: {metadata['brand']}\n"
                response += f"   价格: ¥{metadata['price']}\n"
                response += f"   类别: {metadata['primary_category']}\n"
                response += f"   评分: {metadata['avg_rating']} ({metadata['review_count']}条评论)\n"
                response += f"   产品ID: {product_id}\n"
                response += f"   相关度: {relevance:.2f}\n\n"

            print(f"已记录 {len(self.recommended_products)} 个推荐产品")
            print(f"序号映射: {self.product_order_map}")
            return response
        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            print(f"搜索产品时出错: {str(e)}\n{trace}")
            return f"搜索产品时出错: {str(e)}"

    def _generate_chinese_name(self, english_title, price):
        """从英文标题生成可能的中文名称"""
        # 简单关键词映射示例
        keywords = {
            "dress": "连衣裙",
            "skirt": "裙子",
            "convertible": "可转换",
            "tie dye": "扎染",
            "gypsy": "波西米亚",
            "boho": "波西米亚",
            "summer": "夏季",
            "maxi": "长裙",
            "floral": "花卉",
            "pattern": "图案",
            "empire": "帝国式",
            "style": "风格",
            "women": "女士",
            "stripe": "条纹",
            "sleek": "简约",
            "racerback": "挖背",
            "v-back": "V领",
            "chiffon": "雪纺",
            "lace": "蕾丝"
        }

        # 尝试匹配关键词，生成中文名称
        chinese_parts = []
        title_lower = english_title.lower()

        for eng, chn in keywords.items():
            if eng in title_lower:
                chinese_parts.append(chn)

        if not chinese_parts:
            return None

        # 生成基本中文名称
        chinese_name = "".join(chinese_parts)

        # 添加价格信息，使名称更具体
        chinese_name_with_price = f"【{chinese_name}】¥{price}"

        return chinese_name_with_price

    def find_product_by_partial_name(self, partial_name: str):
        """通过部分名称查找产品（增强版）"""
        if not partial_name:
            return None

        partial_name = partial_name.strip().lower()

        # 首先检查翻译后的中文名称映射
        for name, product_id in self.translated_names.items():
            if partial_name in name.lower():
                print(f"从中文名称映射中找到: {name} -> {product_id}")
                return {"id": product_id, "metadata": self.recommended_products[product_id]["metadata"],
                        "content": self.recommended_products[product_id]["content"]}

        # 1. 尝试精确匹配
        for product_name, product_info in self.recommended_products_by_name.items():
            if partial_name == product_name:
                return product_info

        # 2. 尝试包含匹配
        best_match = None
        highest_similarity = 0

        for product_name, product_info in self.recommended_products_by_name.items():
            # 如果部分名称是产品名的子字符串
            if partial_name in product_name:
                similarity = len(partial_name) / len(product_name)
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = product_info

        # 3. 尝试关键词匹配
        if not best_match:
            partial_words = set(partial_name.split())
            for product_name, product_info in self.recommended_products_by_name.items():
                product_words = set(product_name.split())
                common_words = partial_words.intersection(product_words)
                if common_words:
                    similarity = len(common_words) / len(partial_words)
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        best_match = product_info

        return best_match if highest_similarity > 0.2 else None

    def get_product_details(self, product_id_or_name: str) -> str:
        """获取产品详细信息的工具，增强版"""
        try:
            # 清理产品ID或名称（移除可能的空格和引号）
            product_id_or_name = product_id_or_name.strip().strip('"\'')
            print(f"正在查找产品: '{product_id_or_name}'")

            # 检查是否匹配"第N款"格式
            import re
            ordinal_match = re.match(r'第(\d+)款', product_id_or_name)
            if ordinal_match:
                index = int(ordinal_match.group(1))
                if index in self.product_order_map:
                    product_id = self.product_order_map[index]
                    print(f"从序号映射找到第{index}款商品ID: {product_id}")
                    return self.get_product_details(product_id)  # 递归调用自身，使用ID查询

            # 步骤1: 首先检查是否是之前推荐过的产品ID
            if product_id_or_name in self.recommended_products:
                print(f"从推荐列表中找到产品ID: {product_id_or_name}")
                product_info = self.recommended_products[product_id_or_name]
                metadata = product_info["metadata"]
                content = product_info["content"]

                # 使用推荐商品信息构建响应
                response = self._format_product_details(metadata, content)
                return response

            # 步骤1.5: 检查翻译后的中文名称映射
            for name, product_id in self.translated_names.items():
                if product_id_or_name.lower() in name.lower():
                    print(f"从中文名称映射中找到: {name} -> {product_id}")
                    if product_id in self.recommended_products:
                        product_info = self.recommended_products[product_id]
                        metadata = product_info["metadata"]
                        content = product_info["content"]
                        response = self._format_product_details(metadata, content)
                        return response

            # 步骤2: 尝试按名称查找之前推荐过的产品
            product_match = self.find_product_by_partial_name(product_id_or_name)
            if product_match:
                print(f"通过名称匹配找到产品: {product_match['metadata']['title']}")
                metadata = product_match["metadata"]
                content = product_match["content"]

                # 使用推荐商品信息构建响应
                response = self._format_product_details(metadata, content)
                return response

            # 步骤3: 尝试使用向量存储查找产品
            try:
                print(f"尝试通过向量搜索查找: {product_id_or_name}")
                # 首先尝试直接使用ID搜索
                direct_id_results = self.kb.search_products(f"ID:{product_id_or_name}", k=1)

                # 然后尝试使用关键词搜索
                keyword_results = self.kb.search_products(product_id_or_name, k=3)

                # 合并结果，优先使用ID搜索的结果
                results = direct_id_results + keyword_results if direct_id_results else keyword_results

                if results and results[0]["relevance_score"] > 0.5:
                    # 从向量存储结果中获取产品信息
                    metadata = results[0]["metadata"]
                    content = results[0]["content"]

                    # 保存到推荐产品列表中以便后续引用
                    self.recommended_products[metadata['id']] = {
                        "metadata": metadata,
                        "content": content
                    }
                    self.recommended_products_by_name[metadata['title'].lower()] = {
                        "id": metadata['id'],
                        "metadata": metadata,
                        "content": content
                    }

                    # 生成可能的中文名称并保存映射
                    chinese_name = self._generate_chinese_name(metadata['title'], metadata['price'])
                    if chinese_name:
                        self.translated_names[chinese_name] = metadata['id']

                    # 格式化并返回产品详情
                    response = self._format_product_details(metadata, content)
                    return response
            except Exception as vector_error:
                print(f"向量搜索错误: {str(vector_error)}")
                # 继续尝试其他方法

            # 步骤4: 尝试通过文本名称搜索
            try:
                print(f"尝试通过名称搜索: {product_id_or_name}")
                search_results = self.kb.search_products(product_id_or_name, k=3)
                if search_results:
                    # 获取最相关的结果
                    best_result = search_results[0]
                    metadata = best_result["metadata"]
                    content = best_result["content"]

                    # 保存到推荐列表
                    self.recommended_products[metadata['id']] = {
                        "metadata": metadata,
                        "content": content
                    }
                    self.recommended_products_by_name[metadata['title'].lower()] = {
                        "id": metadata['id'],
                        "metadata": metadata,
                        "content": content
                    }

                    # 生成可能的中文名称并保存映射
                    chinese_name = self._generate_chinese_name(metadata['title'], metadata['price'])
                    if chinese_name:
                        self.translated_names[chinese_name] = metadata['id']

                    # 添加一个提示，说明这可能不是用户最初请求的产品
                    response = f"我找不到精确匹配的'{product_id_or_name}'，但找到了可能相关的产品:\n\n"
                    response += self._format_product_details(metadata, content)
                    return response
            except Exception as search_error:
                print(f"文本搜索错误: {str(search_error)}")

            # 如果所有方法都失败，提供有帮助的错误消息
            return f"抱歉，我无法找到'{product_id_or_name}'的详细信息。请尝试使用更精确的产品名称或ID，或者重新搜索相关产品。"

        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            print(f"获取产品详情时出错: {str(e)}\n{trace}")
            return f"获取产品详情时出错: {str(e)}"

    def _format_product_details(self, metadata, content):
        """格式化产品详情（新增辅助函数）"""
        response = f"产品详情 - {metadata['title']}\n\n"
        response += f"ID: {metadata['id']}\n"
        response += f"品牌: {metadata['brand']}\n"
        response += f"价格: ¥{metadata['price']}\n"
        response += f"类别: {metadata.get('primary_category', '未分类')}\n"
        response += f"评分: {metadata['avg_rating']} ({metadata['review_count']}条评论)\n\n"

        # 从内容中提取描述
        if "描述:" in content:
            desc_parts = content.split("描述:")[1].split("\n")[0].strip()
            response += f"描述: {desc_parts}\n\n"
        else:
            response += "描述: 暂无\n\n"

        # 提取评论
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

    def filter_products_by_price(self, price_range: str) -> str:
        """按价格范围筛选产品的工具 - 改进版"""
        try:
            # 解析价格范围
            parts = price_range.split(',')
            if len(parts) != 2:
                return "价格范围格式不正确，请使用逗号分隔最低价格和最高价格，例如：'100,200'"

            min_price = float(parts[0].strip())
            max_price = float(parts[1].strip())

            # 筛选有效产品（价格在范围内，且有标题的产品）
            filtered_products = [
                p for p in self.kb.products
                if min_price <= p.price <= max_price
                   and p.title.strip() != ""  # 确保产品有标题
            ]

            if not filtered_products:
                return f"未找到价格在 ¥{min_price} - ¥{max_price} 范围内的产品。"

            # 按价格从低到高排序
            filtered_products.sort(key=lambda p: p.price)

            response = f"价格在 ¥{min_price} - ¥{max_price} 范围内的产品：\n\n"

            # 清空之前的推荐结果（因为这是新的搜索）
            self.recommended_products = {}
            self.recommended_products_by_name = {}
            self.product_order_map = {}
            self.translated_names = {}

            for i, product in enumerate(filtered_products[:12], 1):  # 最多显示12个结果
                # 构建产品内容
                product_content = f"标题: {product.title}\n品牌: {product.brand}\n价格: {product.price}\n类别: {product.categories}\n"

                # 保存到推荐列表中
                metadata = {
                    "id": product.id,
                    "title": product.title,
                    "brand": product.brand,
                    "price": product.price,
                    "primary_category": product.primary_category or "未分类",
                    "avg_rating": product.avg_rating,
                    "review_count": product.review_count
                }

                self.recommended_products[product.id] = {
                    "metadata": metadata,
                    "content": product_content
                }

                self.recommended_products_by_name[product.title.lower()] = {
                    "id": product.id,
                    "metadata": metadata,
                    "content": product_content
                }

                # 保存序号到产品ID的映射
                self.product_order_map[i] = product.id

                # 生成可能的中文名称并保存映射
                chinese_name = self._generate_chinese_name(product.title, product.price)
                if chinese_name:
                    self.translated_names[chinese_name] = product.id
                    # 为第N款格式生成映射
                    self.translated_names[f"第{i}款"] = product.id

                response += f"{i}. {product.title}\n"
                response += f"   品牌: {product.brand}\n"
                response += f"   价格: ¥{product.price}\n"
                response += f"   类别: {product.primary_category if product.primary_category else product.categories}\n"
                response += f"   ID: {product.id}\n\n"

            if len(filtered_products) > 12:
                response += f"(共找到 {len(filtered_products)} 个结果，只显示前12个)\n"

            print(f"已记录 {len(self.recommended_products)} 个推荐产品")
            print(f"序号映射: {self.product_order_map}")
            return response
        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            print(f"按价格筛选产品时出错: {str(e)}\n{trace}")
            return f"按价格筛选产品时出错: {str(e)}"



    def analyze_product_reviews(self, product_id: str) -> str:
        """分析产品评论的工具"""
        try:
            # 首先检查是否在推荐产品列表中
            if product_id in self.recommended_products:
                product_info = self.recommended_products[product_id]
                content = product_info["content"]
                metadata = product_info["metadata"]

                # 检查是否有评论信息
                if "评论:" in content:
                    response = f"产品 '{metadata['title']}' 评论分析\n\n"
                    response += f"评分: {metadata['avg_rating']} ({metadata['review_count']}条评论)\n\n"

                    # 提取评论部分
                    review_section = content.split("评论:")[1]
                    review_lines = [line for line in review_section.split("\n") if line.strip()]

                    if review_lines:
                        response += "评论详情:\n"
                        for i, line in enumerate(review_lines, 1):
                            if line.startswith("- 评分:"):
                                response += f"{i}. {line[2:]}\n"  # 移除"- "前缀

                        return response
                    else:
                        return f"产品 '{metadata['title']}' 暂无详细评论信息。"
                else:
                    return f"产品 '{metadata['title']}' 暂无评论。"

            # 检查是否匹配中文名称
            for name, pid in self.translated_names.items():
                if product_id.lower() in name.lower():
                    print(f"从中文名称映射中找到评论: {name} -> {pid}")
                    return self.analyze_product_reviews(pid)  # 递归调用自身

            # 检查是否匹配"第N款"格式
            import re
            ordinal_match = re.match(r'第(\d+)款', product_id)
            if ordinal_match:
                index = int(ordinal_match.group(1))
                if index in self.product_order_map:
                    return self.analyze_product_reviews(self.product_order_map[index])  # 递归调用自身

            # 如果不在推荐列表中，尝试从产品列表中查找
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

            # 如果通过ID找不到，尝试通过名称查找
            product_match = self.find_product_by_partial_name(product_id)
            if product_match:
                metadata = product_match["metadata"]
                content = product_match["content"]

                response = f"产品 '{metadata['title']}' 评论分析\n\n"

                if "评论:" in content:
                    review_section = content.split("评论:")[1]
                    review_lines = [line for line in review_section.split("\n") if line.strip()]

                    if review_lines:
                        response += "评论详情:\n"
                        for i, line in enumerate(review_lines, 1):
                            if line.startswith("- 评分:"):
                                response += f"{i}. {line[2:]}\n"

                        return response
                    else:
                        return f"产品 '{metadata['title']}' 暂无详细评论信息。"
                else:
                    return f"产品 '{metadata['title']}' 暂无评论。"

            return f"未找到ID为'{product_id}'的产品。请尝试使用搜索功能查找相关产品。"
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
                description="获取特定产品的详细信息和评论分析。输入应该是产品ID或产品名称。"
            ),
            Tool(
                name="按价格筛选产品",
                func=self.filter_products_by_price,
                description="按价格范围筛选产品。输入应该是两个数字，用逗号分隔，表示最低价格和最高价格。例如：'10,50'表示10元到50元之间的产品。"
            ),
            Tool(
                name="分析产品评论",
                func=self.analyze_product_reviews,
                description="分析特定产品的评论，提取关键观点和情感。输入应该是产品ID或产品名称。"
            )
        ]