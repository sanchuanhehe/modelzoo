import json
import os
import sys

if __name__ == "__main__":
    # 初始化参数变量：默认不递归，目录参数为空
    recursive = False
    file_dir = None

    # 解析命令行参数（支持两种输入格式：1. 仅目录 2. -r + 目录）
    if len(sys.argv) == 2:
        # 情况1：仅传入目录（无递归参数）
        file_dir = sys.argv[1]
    elif len(sys.argv) == 3 and sys.argv[1] == "-r":
        # 情况2：传入 -r + 目录（开启递归）
        recursive = True
        file_dir = sys.argv[2]
    else:
        # 参数格式错误，提示正确用法
        print("用法错误！正确格式：")
        print("1. 非递归查找：python generate_file_list.py [目录路径]")
        print("2. 递归查找：python generate_file_list.py -r [目录路径]")
        exit(1)

    # 验证目录参数是否存在
    if not file_dir:
        print("empty input")
        exit(1)

    # 验证目录是否真实存在
    if not os.path.exists(file_dir):
        print("dir does not exist. path: {}".format(file_dir))
        exit(1)

    file_path_list = []

    if recursive:
        # 递归模式：遍历目录及其所有子目录
        # os.walk 生成三元组 (当前目录, 子目录列表, 文件列表)
        for root, dirs, files in os.walk(file_dir):
            # 排序文件列表，保证输出顺序一致
            files.sort()
            for file_name in files:
                # 过滤 .bin 后缀文件
                if file_name.endswith(".bin"):
                    continue
                # 拼接完整路径（root是当前文件所在的目录路径）
                full_path = os.path.join(root, file_name)
                # 构建上级目录相对路径（与非递归模式路径格式保持一致）
                relative_path = "../" + full_path
                # 添加到路径列表（保持二维列表格式）
                file_path_list.append([relative_path])
    else:
        # 非递归模式：保持原有逻辑，仅遍历当前目录
        file_name_list = os.listdir(file_dir)
        file_name_list.sort()
        for file_name in file_name_list:
            # 过滤 .bin 后缀文件和非文件（文件夹）
            if file_name.endswith(".bin") or not os.path.isfile(os.path.join(file_dir, file_name)):
                continue
            # 构建上级目录相对路径
            file_path = "../" + os.path.join(file_dir, file_name)
            file_path_list.append([file_path])

    # 构建JSON数据结构
    json_data = {"fileList": file_path_list}

    # 确保 data 目录存在（避免写入时报错）
    os.makedirs("data", exist_ok=True)

    # 将数据写入JSON文件
    with open("data/file_list.json", "w") as f:
        json.dump(json_data, f, indent=4)

    # 打印执行结果（区分递归/非递归模式）
    mode = "递归" if recursive else "非递归"
    print("{}查找完成！JSON文件已生成到data/file_list.json，共包含{}个文件路径。".format(mode, len(file_path_list)))
