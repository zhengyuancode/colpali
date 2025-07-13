import subprocess

def run_mineru(input_path: str, output_path: str):
    """
    调用 mineru 命令行工具，将指定的输入文件处理后保存到输出文件。

    参数:
        input_path (str): 输入文件的路径。
        output_path (str): 输出文件的路径。
    
    返回:
        subprocess.CompletedProcess: 执行结果对象。
    
    异常:
        subprocess.CalledProcessError: 如果命令执行失败。
    """
    command = ['/etc/anaconda3/envs/colpali/bin/mineru', '-p', input_path, '-o', output_path, '--source', 'local']
    
    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.stderr:
            print("警告或错误信息：")
            print(result.stderr)
        return result
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败，错误码 {e.returncode}")
        print(f"错误信息：\n{e.stderr}")
        raise

if __name__ == "__main__":
    input_file = "example_input.txt"
    output_file = "example_output.txt"
    
    try:
        run_mineru(input_file, output_file)
        print("mineru 处理完成。")
    except subprocess.CalledProcessError:
        print("mineru 处理失败，请检查输入和环境配置。")