import json
import random
from collections import defaultdict


def process_metadata(metadata_file, output_file):

    try:
        print(f"正在读取元数据文件: {metadata_file}")

        # 逐行读取JSONL文件
        processed_metadata = {}
        line_count = 0

        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:

                    item = eval(line)
                    line_count += 1

                    # 确保有asin字段
                    if 'asin' not in item:
                        continue

                    # 创建新的商品数据，排除related和imUrl字段
                    product = {
                        "asin": item["asin"],
                        "title": item.get("title", ""),
                        "price": item.get("price", 0.0),
                        "salesRank": item.get("salesRank", {}),
                        "brand": item.get("brand", ""),
                        "categories": item.get("categories", [])
                    }

                    # 添加描述字段（如果存在）
                    if "description" in item:
                        product["description"] = item["description"]

                    # 保存处理后的商品数据
                    processed_metadata[item["asin"]] = product

                    # 显示处理进度
                    if line_count % 10000 == 0:
                        print(f"已处理 {line_count} 行元数据")

                except Exception as e:
                    print(f"警告: 跳过无效的行: {line[:100]}..., 错误: {str(e)}")

        print(f"成功处理 {len(processed_metadata)} 条商品元数据，共读取 {line_count} 行")

        # 保存处理后的元数据
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(list(processed_metadata.values()), f, indent=2, ensure_ascii=False)

        print(f"处理后的元数据已保存至 {output_file}")
        return processed_metadata

    except Exception as e:
        print(f"处理元数据文件出错: {str(e)}")
        return {}


def process_reviews(reviews_file, output_file, max_reviews=15):

    try:
        print(f"正在读取评论文件: {reviews_file}")


        reviews_dict = defaultdict(list)
        line_count = 0


        with open(reviews_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    # 解析每行的JSON对象
                    review = eval(line)
                    line_count += 1

                    # 确保有asin字段
                    if 'asin' not in review:
                        continue

                    # 创建新的评论数据，仅保留需要的字段
                    processed_review = {
                        "asin": review["asin"],
                        "helpful": review.get("helpful", [0, 0]),
                        "reviewText": review.get("reviewText", ""),
                        "overall": review.get("overall", 0.0),
                        "summary": review.get("summary", "")
                    }

                    # 将处理后的评论添加到对应商品的评论列表
                    reviews_dict[review["asin"]].append(processed_review)

                    # 显示处理进度
                    if line_count % 10000 == 0:
                        print(f"已处理 {line_count} 行评论")

                except Exception as e:
                    print(f"警告: 跳过无效的行: {line[:100]}..., 错误: {str(e)}")

        # 对每个商品的评论进行随机抽样
        for asin, reviews in reviews_dict.items():
            if len(reviews) > max_reviews:
                reviews_dict[asin] = random.sample(reviews, max_reviews)

        print(
            f"成功处理评论，涉及 {len(reviews_dict)} 种商品，共读取 {line_count} 行，每个商品随机抽取最多 {max_reviews} 条评论")

        # 保存处理后的评论
        with open(output_file, 'w', encoding='utf-8') as f:
           
            all_reviews = []
            for asin_reviews in reviews_dict.values():
                all_reviews.extend(asin_reviews)
            json.dump(all_reviews, f, indent=2, ensure_ascii=False)

        print(f"处理后的评论已保存至 {output_file}")
        return reviews_dict

    except Exception as e:
        print(f"处理评论文件出错: {str(e)}")
        return {}


def calculate_avg_rating(reviews):
    """计算平均评分"""
    if not reviews:
        return 0.0
    total = sum(review["overall"] for review in reviews)
    return round(total / len(reviews), 1)


def merge_data(metadata, reviews, output_file):
    """合并商品数据和评论数据，剔除价格为0和评论数为0的商品"""
    try:
        print("正在合并商品数据和评论数据...")
        # 创建合并后的知识库
        knowledge_base = []

        # 统计数据
        total_products = len(metadata)
        filtered_price_zero = 0
        filtered_no_reviews = 0
        valid_products = 0

        # 处理每个商品
        count = 0
        for asin, product in metadata.items():
            # 获取该商品的评论
            product_reviews = reviews.get(asin, [])
            avg_rating = calculate_avg_rating(product_reviews)
            review_count = len(product_reviews)

            # 跳过价格为0的商品
            if product.get("price", 0.0) <= 0:
                filtered_price_zero += 1
                continue

            # 跳过评论数为0的商品
            if review_count == 0:
                filtered_no_reviews += 1
                continue

            # 创建知识库文档
            doc = {
                "id": asin,
                "title": product.get("title", ""),
                "brand": product.get("brand", ""),
                "price": product.get("price", 0.0),
                "categories": product.get("categories", []),
                "description": product.get("description", ""),
                "reviews": product_reviews,
                "review_count": review_count,
                "avg_rating": avg_rating
            }

            # 添加销售排名信息
            sales_rank = product.get("salesRank", {})
            if sales_rank:
                primary_category = next(iter(sales_rank.keys()), "")
                doc["primary_category"] = primary_category
                doc["sales_rank"] = sales_rank.get(primary_category, 0)

            # 将有效商品添加到知识库
            knowledge_base.append(doc)
            valid_products += 1

            count += 1
            if count % 10000 == 0:
                print(f"已处理 {count} 个商品")

        # 输出统计信息
        print(f"原始商品总数: {total_products}")
        print(f"剔除价格为0的商品: {filtered_price_zero}")
        print(f"剔除评论数为0的商品: {filtered_no_reviews}")
        print(f"保留的有效商品数: {valid_products}")
        print(f"知识库构建完成，包含 {len(knowledge_base)} 个商品文档")

        # 保存知识库
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(knowledge_base, f, indent=2, ensure_ascii=False)

        print(f"合并后的知识库已保存至 {output_file}")

    except Exception as e:
        print(f"合并数据出错: {str(e)}")


def main():
    # 设置随机种子以获得可重现的结果
    random.seed(42)

    # 输入文件路径
    metadata_json = "data/meta_Clothing_Shoes_and_Jewelry.json"#商品元数据
    reviews_json = "data/reviews_Clothing_Shoes_and_Jewelry.json"#商品评论数据


    output_dir = "E:/agent/AgentShop/new_data/"
    processed_metadata_json = output_dir + "processed_metadata.json"
    processed_reviews_json = output_dir + "processed_reviews.json"

    # 最终输出文件
    output_json = output_dir + "product_knowledge_base.json"

    # 处理元数据
    metadata = process_metadata(metadata_json, processed_metadata_json)

    # 处理评论数据，每个商品最多保留50条随机评论
    reviews = process_reviews(reviews_json, processed_reviews_json, max_reviews=50)

    # 合并数据，剔除价格为0和评论数为0的商品
    merge_data(metadata, reviews, output_json)

    print("数据处理完成！最终知识库文件保存在：" + output_json)


if __name__ == "__main__":
    main()
