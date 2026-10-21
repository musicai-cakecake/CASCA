import pickle
import numpy as np
import os

class IndexedDatasetReader:
    """配套的读取类：对应之前的 IndexedDatasetBuilder"""

    def __init__(self, base_path):
        """
        初始化读取器
        :param base_path: 文件基础路径（不带后缀），你的路径：data/test_out/valid
        """
        self.base_path = base_path
        self.data_path = f"{base_path}.data"  # .data 数据文件
        self.idx_path = f"{base_path}.idx"  # .idx 索引文件
        self.length_path = f"{base_path}_length.npy"  # 总数文件

        # 1. 加载索引（字节偏移量）
        if not os.path.exists(self.idx_path):
            raise FileNotFoundError(f"索引文件不存在：{self.idx_path}")
        idx_data = np.load(self.idx_path, allow_pickle=True).item()
        self.offsets = idx_data['offsets']  # 读取偏移量列表
        self.total_num = len(self.offsets) - 1  # 数据总数（偏移量长度-1）

        # 2. 打开数据文件（二进制只读模式）
        self.data_file = open(self.data_path, 'rb')

    def get_total_count(self):
        """功能1：读取有效数据总数"""
        # 方式1：从索引计算（推荐）
        # return self.total_num
        # 方式2：直接读取_length.npy文件
        return np.load(self.length_path).item()

    def get_item_by_idx(self, idx):
        """功能2：根据指定索引读取单条数据
        :param idx: 数据索引（从 0 开始）
        """
        # 校验索引合法性
        if idx < 0 or idx >= self.total_num:
            raise IndexError(f"索引超出范围！有效索引：0 ~ {self.total_num - 1}，你输入的：{idx}")

        # 定位字节位置：当前数据的起始偏移量
        start_offset = self.offsets[idx]
        # 结束偏移量（下一个数据的起始位置）
        end_offset = self.offsets[idx + 1]
        # 计算需要读取的字节长度
        read_bytes = end_offset - start_offset

        # 跳转到指定字节位置
        self.data_file.seek(start_offset)
        # 读取字节数据
        data_bytes = self.data_file.read(read_bytes)
        # 反序列化为Python对象
        item = pickle.loads(data_bytes)
        return item

    def get_all_items(self):
        """功能3：读取所有数据，返回Python列表（所有数据对象）"""
        all_items = []
        for idx in range(self.total_num):
            item = self.get_item_by_idx(idx)
            all_items.append(item)
        return all_items

    def close(self):
        """关闭文件（必须调用，释放资源）"""
        if hasattr(self, 'data_file') and not self.data_file.closed:
            self.data_file.close()


def count_empty_token_ratio(reader, target_token: float):
    total_data_num = reader.total_num
    total_tokens = 0  # 所有维度的Token总数
    empty_tokens = 0  # 目标浮点数Token总数

    print(f"开始统计 二维列表 浮点数 token={target_token} (empty) 占比...")

    for idx in range(total_data_num):
        item = reader.get_item_by_idx(idx)

        # --------------------------
        # 你的数据：item = 二维token列表 (如 [[1.0, 1829.0], [2.0, 3.0, ...]])
        token_seq = item
        # 如果是字典格式：token_seq = item['tokens']
        # --------------------------

        # 兼容：二维numpy数组 转 二维列表
        if isinstance(token_seq, np.ndarray):
            token_seq = token_seq.tolist()

        # ============== 遍历【二维列表】核心代码 ==============
        # 遍历每一个一维子序列
        for sub_seq in token_seq:
            # 遍历子序列中的每一个浮点数token
            for token in sub_seq:
                total_tokens += 1  # 总Token数+1
                # 方案1：精确匹配浮点数（推荐，你的token是标准1829.0）
                if token == target_token:
                    empty_tokens += 1
                # 方案2：浮点精度近似匹配（有误差时打开，二选一）
                # if math.isclose(token, target_token, abs_tol=1e-6):
                #     empty_tokens += 1

        # 进度打印
        if (idx + 1) % 1000 == 0:
            print(f"已处理 {idx + 1}/{total_data_num} 条数据")

    # 计算占比
    ratio = (empty_tokens / total_tokens) * 100 if total_tokens > 0 else 0.0
    return total_data_num, total_tokens, empty_tokens, ratio


# ===================== 核心：多目标Token + 二维列表 + 浮点数 =====================
def count_multi_tokens_ratio(reader, target_tokens: list):
    """
    统计多个目标Token的占比（支持浮点数、二维列表）
    :param reader: 数据读取器
    :param target_tokens: 目标Token列表，如 [1829.0, 15.0]
    :return: 统计结果
    """
    total_data_num = reader.total_num
    total_tokens = 0  # 所有Token总数
    # 初始化每个目标Token的计数器
    token_counts = {token: 0 for token in target_tokens}
    total_matched = 0  # 所有目标Token的总匹配数

    print(f"开始统计 多目标Token：{target_tokens}")
    print("=" * 70)

    for idx in range(total_data_num):
        item = reader.get_item_by_idx(idx)

        # 你的数据：二维Token列表
        token_seq = item
        # 字典格式：token_seq = item['tokens']

        # 兼容numpy数组转列表
        if isinstance(token_seq, np.ndarray):
            token_seq = token_seq.tolist()

        # 遍历二维列表所有元素
        for sub_seq in token_seq:
            for token in sub_seq:
                total_tokens += 1
                # 匹配所有目标Token（浮点数精确匹    配）
                for target in target_tokens:
                    # ====================== 匹配规则 ======================
                    # 方案1：精确匹配（推荐，无浮点误差用这个）
                    if token == target:
                        token_counts[target] += 1
                        total_matched += 1
                        break  # 一个token只匹配一个目标，避免重复计数

                    # 方案2：浮点近似匹配（有精度误差用这个，二选一）
                    # if math.isclose(token, target, abs_tol=1e-6):
                    #     token_counts[target] += 1
                    #     total_matched += 1
                    #     break

        # 进度打印
        if (idx + 1) % 1000 == 0:
            print(f"已处理 {idx + 1}/{total_data_num} 条数据")

    # 计算占比
    token_ratios = {t: (cnt / total_tokens) * 100 for t, cnt in token_counts.items()}
    total_ratio = (total_matched / total_tokens) * 100 if total_tokens > 0 else 0.0

    return total_data_num, total_tokens, token_counts, token_ratios, total_matched, total_ratio


# ------------------- 以下是使用示例（直接运行即可） -------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inspect externally prepared indexed music data")
    parser.add_argument("base_path", help="Dataset split prefix, without .data/.idx suffix")
    args = parser.parse_args()
    BASE_PATH = args.base_path


    # 初始化读取器
    reader = IndexedDatasetReader(BASE_PATH)

    try:
        # ============== 功能1：读取有效数据总数 ==============
        total_count = reader.get_total_count()
        print(f"✅ 有效数据总数：{total_count}")

        # ============== 功能2：根据指定索引读取一条数据 ==============
        target_idx = 0  # 你可以修改为任意索引：0,1,2...
        single_item = reader.get_item_by_idx(target_idx)
        print(f"\n✅ 索引 {target_idx} 的数据：")
        print(f"数据类型：{type(single_item)}")
        print(f"数据内容：{single_item}")
        print(f"数据维度：{single_item.shape}")

        # ============== 功能3：读取所有数据到Python对象 ==============
        all_data = reader.get_all_items()
        print(f"\n✅ 所有数据读取完成！总条数：{len(all_data)}")
        # 可直接使用 all_data 列表（里面是你之前保存的所有item对象）

        # 执行统计
        target_tokens = [605] # pad的idx
        total_data, total_tok, counts, ratios, total_match, total_ratio = count_multi_tokens_ratio(
            reader, target_tokens
        )

        # -------------- 打印详细结果 --------------
        print("\n" + "=" * 70)
        print(f"📊 多目标Token统计完成（二维列表+浮点数）")
        print(f"总数据条数：{total_data}")
        print(f"全量总Token数：{total_tok}")
        print("-" * 70)

        # 打印每个Token的统计结果
        for token in target_tokens:
            print(f"Token {token} | 数量：{counts[token]:>8} | 占比：{ratios[token]:>6.4f}%")

        print("-" * 70)
        # 打印汇总结果
        print(f"所有目标Token总数量：{total_match}")
        print(f"所有目标Token总占比：{total_ratio:.4f}%")

    finally:
        # 必须关闭文件
        reader.close()
