import json
from pathlib import Path
from typing import Any, Dict, List, Union


def get_data_structure(obj: Any, indent: int = 0) -> str:
    """
    递归解析数据结构描述，不输出实际数据内容
    """
    prefix = "  " * indent
    if isinstance(obj, dict):
        lines = [f"{prefix}dict{{}}"]
        for k, v in obj.items():
            sub_struct = get_data_structure(v, indent + 1)
            lines.append(f"{prefix}  '{k}': {sub_struct.lstrip()}")
        return "\n".join(lines)
    elif isinstance(obj, list):
        if len(obj) > 0:
            sample_type = get_data_structure(obj[0], 0).splitlines()[0]
            return f"list[ {sample_type} ]"
        else:
            return "list[空]"
    elif isinstance(obj, str):
        return "str"
    elif isinstance(obj, int):
        return "int"
    elif isinstance(obj, float):
        return "float"
    elif isinstance(obj, bool):
        return "bool"
    elif obj is None:
        return "None"
    else:
        return type(obj).__name__


def show_first_jsonl_structure(file_path: Union[str, Path]):
    """
    读取超大jsonl，只读取第一行，分析字典嵌套结构，不打印数据内容
    """
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"文件不存在：{file_path}")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        first_line = None
        # 只读取第一行，不会加载整个大文件
        for line in f:
            line = line.strip()
            if line:
                first_line = line
                break

    if first_line is None:
        print("文件为空或没有有效行")
        return

    try:
        data: Dict[str, Any] = json.loads(first_line)
    except Exception as e:
        print(f"第一行json解析失败：{e}")
        return

    print("===== JSONL 第一行数据结构（仅结构，无真实值）=====")
    struct_text = get_data_structure(data)
    print(struct_text)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inspect a JSONL record without loading the full file")
    parser.add_argument("jsonl", type=Path)
    args = parser.parse_args()
    show_first_jsonl_structure(args.jsonl)