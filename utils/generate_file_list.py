import json
import os
import sys

if __name__ == "__main__":
    try:
        file_dir = sys.argv[1]  # 输入文件目录
    except IndexError:
        print("empty input")
        exit(1)

    if not (os.path.exists(file_dir)):
        print("dir does not exist. path: {}".format(file_dir))
        exit(1)

    file_path_list = []
    file_name_list = os.listdir(file_dir)
    file_name_list.sort()

    for file_name in file_name_list:
        if file_name.endswith(".bin") or not os.path.isfile(
            os.path.join(file_dir, file_name)
        ):
            continue
        file_path = "../" + os.path.join(
            file_dir, file_name
        )  # out文件夹相对于图片相对地址
        file_path_list.append([file_path])

    # 构建JSON数据结构
    json_data = {"fileList": file_path_list}

    # 将数据写入JSON文件
    with open("data/file_list.json", "w") as f:
        json.dump(json_data, f, indent=4)

    print("JSON文件生成完成！共包含{}个文件路径。".format(len(file_path_list)))
