import subprocess
import json
import os
import sys
import time
import stat
import shutil
import hashlib
import tarfile
from typing import List, Dict, Union, Set, Optional

# 编译配置
BUILD_INFO_FILENAME = 'build_config.json'
DEFAULT_BUILD_TIMEOUT = 60 * 15

def get_local_branches() -> List[str]:
    """获取所有本地分支列表"""
    result = subprocess.run(
        ["git", "branch", "--format=%(refname:short)"],
        capture_output=True,
        text=True
    )
    return result.stdout.strip().split("\n") if result.stdout else []

def check_changes_and_get_folders(changed_files: List[str]) -> Optional[Set[str]]:
    # 定义要剔除的文件和目录
    files_to_remove = ['samples/build_sample.py','samples/build_gate.sh','samples/build_script.sh']
    directories_to_remove = [
        'samples/common',
        'samples/samples_GPL/common',
        'samples/acl_utils/include/common',
        'samples/samples_GPL/acl_utils/include/common',
        'samples/opensource'
    ]
    # 剔除匹配的文件和目录
    changed_files = [f for f in changed_files if not (f in files_to_remove or any(f.startswith(d) for d in directories_to_remove))]

    """
    检查变更文件并返回受影响的文件夹集合
    返回None表示不需要构建或存在非法修改
    """
    # 检查是否有代码文件修改
    has_c_or_h_files = any(f.endswith(('.c', '.h', '.cpp', 'txt', '.py')) for f in changed_files)
    
    if not has_c_or_h_files:
        print("Not need build, only non-source files modified")
        sys.exit(0)
    
    # 检查是否修改了src目录或构建脚本
    for file_path in changed_files:
        if '/' in file_path:
            src_folder_name = file_path.split('/')[0]
            if '"src' in src_folder_name or 'src' in src_folder_name:
                print(f"invalid modify, not allow modify src dir and build script")
                print(f"build exit cause")
                sys.exit(0)
                # Check if it's modifying 'build_sample.py' file
            # if 'build_sample.py' in file_path:
            #     print(f"invalid modify, not allow modify build_sample.py")
            #     sys.exit(0)
    # 提取三级文件夹结构
    changed_folders = set()
    for file_path in changed_files:
        if file_path.endswith(('.c', '.h', '.cpp', 'txt', '.py')):
            parts = file_path.split('/')
            if len(parts) >= 4:  # 确保路径深度足够
                folder_name = '+'.join(parts[1:4])  # 取第2-4级目录
                changed_folders.add(folder_name)
    
    print(f"[Changed folders]: {changed_folders}")
    return changed_folders

def compare_with_remote_master() -> Dict[str, Union[List[str], str, Set[str]]]:
    """
    比较所有本地分支与远程master分支的差异
    返回字典格式: {
        分支名: {
            'files': 差异文件列表,
            'folders': 受影响的文件夹集合(仅C/H文件),
            'error': 错误信息(如果有)
        }
    }
    """
    local_branches = get_local_branches()
    remote_master = "origin/master"
    branch_diff = {}

    print(f"Comparing {len(local_branches)} local branches with {remote_master}\n")
    
    for branch in local_branches:
        print(f"🔍 Branch [{branch}] vs {remote_master}:")
        branch_info = {}
        
        # 获取差异文件列表
        diff_result = subprocess.run(
            ["git", "diff", "--name-only", remote_master, branch],
            capture_output=True,
            text=True
        )
        
        if diff_result.returncode != 0:
            error_msg = f"Error: {diff_result.stderr.strip()}"
            print(f"{error_msg}")
            branch_info['error'] = error_msg
            branch_diff[branch] = branch_info
            continue
            
        changed_files = diff_result.stdout.strip().split("\n") if diff_result.stdout else []
        branch_info['files'] = changed_files
        
        if not changed_files:
            print("    Identical to remote master")
        else:
            print(f"{len(changed_files)} changed files:")
            for file in changed_files:
                print(f"- {file}")
            
            # 检查变更并获取文件夹信息
            changed_folders = check_changes_and_get_folders(changed_files)
    return changed_folders


# 获取代码仓所有build_info.json文件内容，并拼接在一起
def process_build_info_files():
    print(f"start process_build_info_files")
    result_list = []
    # 遍历指定目录及其子目录下的所有文件和文件夹
    for root, dirs, files in os.walk("./"):
        for file in files:
            if file == BUILD_INFO_FILENAME:
                file_path = os.path.join(root, file)
                print(file_path)
                # 读取JSON文件内容
                with open(file_path, 'r') as f:
                    try:
                        data = json.load(f)
                        for item in data:
                            # 提取需要的字段值
                            build_target = item.get('buildTarget', '')
                            relative_path = item.get('relativePath', '')
                            chip_name = item.get('chip', '')
                            build_def = item.get('buildDef', '')
                            # 组合成一个字符串并添加到结果列表
                            if build_def:
                                combined_value = f"{file_path}+{build_target}+{relative_path}+{chip_name}+{build_def}"
                            else:
                                combined_value = f"{file_path}+{build_target}+{relative_path}+{chip_name}"
                            result_list.append(combined_value)
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON in file: {file_path}")
                        print(f"build exit cause")
                        sys.exit(-1) 
    return result_list

# 对比git和json内容，一致则代表提交，不一样代表之前提交
def extract_exact_match(input_list, match_list):
    print(f"start extract_exact_match")
    print(f"[extract_exact_match] input_list: {input_list}")
    print(f"[extract_exact_match] match_list: {match_list}")
    exact_matches = []
    try:
        for string in input_list:
            # 使用 split 方法获取第二个 + 和第三个 + 之间的内容
            parts = string.split('+')
            if len(parts) >= 4:
                sample_company_name = parts[0].split('/')[2]
                sample_name_field = parts[2].replace('/','+')
                combined_string = sample_company_name + "+" + sample_name_field
                if combined_string in match_list:
                    exact_matches.append(string)
        print(f"[extract_exact_match] exact_matches: {exact_matches}")
        if not exact_matches:
            print(f"build-config has not been synchronously updated")
            exit(0)  # 退出并返回非零状
        else:
            print("exact_matche 列表不为空")
            return exact_matches
    except TypeError as e:
        print(f"Error: {e}")
        exit(0)

def insert_content_before_line(file_path, target_line, content_to_insert):
    print(f"start insert_content_before_line")
    found_target_line = False
    try:
        # 以读写模式打开文件
        with open(file_path, 'r') as file:
            lines = file.readlines()
            # 查找目标字符串并在其上一行添加列表中第一个内容
            for i in range(len(lines)):
                if target_line in lines[i]:
                    lines.insert(i, content_to_insert)
                    found_target_line = True
                    print(f"成功在文件 {file_path} 中的目标行 {target_line} 的上一行插入内容{content_to_insert}。")
                    break  # 找到目标字符串后添加内容并退出循环

            # 如果没找到目标字符串则直接在该文件最后一行加入内容
            if not found_target_line:
                lines.append(content_to_insert)
                print(f"成功在文件 {file_path} 中的最后一行插入内容{content_to_insert}。")

            with open(file_path, 'w') as file:
                file.writelines(lines)
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
        print(f"build exit cause")
        sys.exit(-1) 

def build_sample(source_directory, chip_name, builddef_engine, log_basic_name, build_target):
    root_dir = '/home/build/modelzoo'
    build_directory_path = 'samples'

    # 检查代码仓目录是否存在
    if not os.path.exists(root_dir):
        print(f"Error: {root_dir} does not exist")
        print(f"build exit cause")
        sys.exit(-1)

    # 创建目录archives
    if not os.path.exists("./archives"):
        os.mkdir("./archives")
    os.chdir(root_dir)
    log_path = os.path.join('.', 'archives', f'build-{log_basic_name}_{chip_name}_{builddef_engine}.log')
    writer = os.fdopen(os.open(
        log_path,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        stat.S_IWUSR | stat.S_IRUSR,
    ), 'wb')
    reader = os.fdopen(os.open(
        log_path,
        os.O_RDONLY,
        stat.S_IWUSR | stat.S_IRUSR,
    ), 'rb')
    os.chmod(log_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    # 切换到目标目录
    os.chdir(build_directory_path)
    build_dir = root_dir + '/' + build_directory_path + '/' + source_directory
    build_gate = root_dir + '/' + build_directory_path + '/build_gate.sh'

    try:
        # 打开日志文件准备写入
        # 使用 subprocess.Popen() 执行命令，并将标准输出和标准错误重定向到日志文件
        process = subprocess.Popen([build_gate, chip_name, builddef_engine, build_dir, build_target], text=False, stdout=writer, stderr=writer)
        # 等待进程结束
        start = time.time()
        while True:
            timeout = (time.time() - start) > DEFAULT_BUILD_TIMEOUT

            line = reader.readline()

            if line == b'':
                if process.poll() is not None:
                    break

                time.sleep(2)
                if not timeout:
                    continue
                else:
                    process.kill()
                    raise Exception(f"build exit cause: timeout")

            try:
                outs = line.decode('utf-8', errors='strict').rstrip()
            except UnicodeDecodeError:
                outs = line.decode('gbk', errors='replace').rstrip()
                print(f"build exit cause")
                sys.exit(-1) 
            if not outs:
                if not timeout:
                    continue
                else:
                    process.kill()
                    raise Exception(f"build timeout")
            print(outs)
        #检查进程的返回码
        if process.returncode == 0:
            # 执行复制操作
            # 源目录路径
            src_file = os.path.join(root_dir, f'{build_directory_path}/{source_directory}/out/main')

            # 检查可执行文件是否存在
            if not os.path.exists(src_file):
                print(f"Error: {src_file} does not exist")
                print(f"build exit cause")
                sys.exit(-1) 
            
            # 目标目录路径
            tar_path = os.path.join(root_dir, f'archives/{log_basic_name}_{chip_name}_{builddef_engine}.tar')
            print(f"src_file: {src_file}")
            print(f"tar_path: {tar_path}")
            try:
                # 修改点2：创建tar压缩包
                with tarfile.open(tar_path, 'w') as tar:
                    # 将可执行文件添加到压缩包，arcname参数控制压缩包内文件名
                    tar.add(src_file, arcname=os.path.basename(src_file))
                print(f"文件已成功打包至 {tar_path}")
            except Exception as e:
                print(f"发生错误: {e}")
                print(f"build exit cause")
                sys.exit(-1) 
            print(f"{source_directory} {chip_name} sample compilation succeeded. Log saved to '{log_path}'.")
        else:
            print(f"{source_directory} {chip_name} sample compilation failed. Check '{log_path}' for details.")
            # 输出标准错误信息到控制台
            print(process.stderr.read().decode('utf-8'))
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print(f"build exit cause")
        sys.exit(-1) 
    if writer:
        writer.close()
    if reader:
        reader.close()

# sample编译前需要添加到编译工具链
def sample_build_prepare(input_list):
    print(f"start sample_build_prepare")
    print(f"[sample_build_prepare] input_list: {input_list}")
    for string in input_list:
        parts = string.split('+')
        sample_file_path = parts[0].split('/')[2]
        build_target = parts[1]
        sample_name = parts[2]
        chip_name = parts[3]
        if len(parts) > 4:
            builddef_engine = parts[4]
        else:
            builddef_engine = 'VOID'

        target_string = sample_file_path.split('/build_config.json')[0] + '/'
        source_directory = target_string + sample_name
        sample_name = sample_name.replace('/','-')
        print(source_directory)
        print(f"[sample_build_prepare] source_directory: {source_directory}")
        log_basic_name = f'{build_target}_{sample_name}'
        build_sample(source_directory, chip_name, builddef_engine, log_basic_name, build_target)

def main():
    print(f"start main")
    check_list = compare_with_remote_master()
    input_list = process_build_info_files()
    sample_name = extract_exact_match(input_list, check_list)
    sample_build_prepare(sample_name)
    print(f"all build step execute end")

if __name__ == '__main__':
    sys.exit(main())
    
