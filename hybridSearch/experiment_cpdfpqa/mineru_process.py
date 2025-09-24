import subprocess

def run_mineru(input_path: str, output_path: str):
    command = ['mineru', '-p', input_path, '-o', output_path, '--source', 'modelscope']
    
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