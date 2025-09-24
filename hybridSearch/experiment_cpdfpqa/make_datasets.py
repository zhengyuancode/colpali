import json
import random
import os
import requests
import fitz
from pathlib import Path
import base64
from openai import OpenAI

QWENAPIKEY="sk-f78b07615c8a45128d760579e6d42e1f"
AIclient = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=QWENAPIKEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def random_sample_jsonl(file_path, k=10):
    """
    从 JSONL 文件中随机抽取 k 行，使用蓄水池抽样算法
    """
    reservoir = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue  # 跳过空行

            # 前 k 行直接放入 reservoir
            if i < k:
                try:
                    obj = json.loads(line)
                    reservoir.append(obj)
                except json.JSONDecodeError:
                    # 解析失败跳过（可选记录）
                    continue
            else:
                # 从 [0, i] 中随机选一个数，如果小于 k，则替换 reservoir 中对应位置
                j = random.randint(0, i)
                if j < k:
                    try:
                        obj = json.loads(line)
                        reservoir[j] = obj
                    except json.JSONDecodeError:
                        # 解析失败，不替换
                        pass
    return reservoir


def download_arxiv_pdf(metadata, outputdir="pdfs"):
    paper_id = metadata['id'] 
    if "/" in paper_id:
        paper_name = paper_id.replace('/', '_')
    else:
        paper_name = paper_id
    pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"

    os.makedirs(outputdir, exist_ok=True)

    filename = f"{paper_name}.pdf"
    filepath = os.path.join(outputdir, filename)

    try:
        print(f"正在从 {pdf_url} 下载...")
        response = requests.get(pdf_url, stream=True, timeout=30)
        response.raise_for_status() 

        content_type = response.headers.get('Content-Type', '')
        if 'application/pdf' not in content_type:
            print(f"警告: 返回的内容类型为 {content_type}，可能不是 PDF。")

        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"成功下载: {filepath}")
        return filepath

    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            print(f"错误: 找不到 PDF（404），论文 {paper_id} 可能不存在或无法生成 PDF。")
        else:
            print(f"HTTP 错误: {e}")
    except requests.exceptions.RequestException as e:
        print(f"下载失败: {e}")
    except Exception as e:
        print(f"保存文件时出错: {e}")

    return None


def pdf_random_pages_to_png(pdf_path, cpdfpqa_json, k=10,dpi=200):
    """
    从 PDF 中随机抽取 k 个页面，保存为 PNG 图像到与 PDF 同名的文件夹中。
    同时生成检索语句json数据集

    参数：
    - pdf_path: PDF 文件路径
    - k: 要抽取的页面数量
    - dpi: 渲染分辨率（影响图像清晰度）
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        print(f"文件不存在: {pdf_path}")
        return
    
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    if k >= total_pages:
        print(f"请求的 k={k} >= 总页数 {total_pages}，将抽取所有页面。")
        selected_indices = list(range(total_pages))
    else:
        selected_indices = random.sample(range(total_pages), k)

    output_folder = pdf_path.parent / pdf_path.stem
    output_folder.mkdir(exist_ok=True)

    print(f"PDF 总页数: {total_pages}")
    print(f"随机选中页面: {[i+1 for i in selected_indices]}")  # 显示为 1-based
    print(f"输出目录: {output_folder}/")

    zoom = dpi / 72  # PDF 默认分辨率为 72 DPI
    mat = fitz.Matrix(zoom, zoom)

    for page_idx in selected_indices:
        page = doc.load_page(page_idx) 
        pix = page.get_pixmap(matrix=mat, alpha=False) 
        img_path = output_folder / f"page_{page_idx + 1}.png"
        pix.save(img_path)
        qi_pair={
            "img_path":str(img_path),
            "query":get_query_by_img(img_path)
        }
        cpdfpqa_json["samples"].append(qi_pair)
        print(f"已保存: {img_path}")

    doc.close()
    
    
    print(f"完成！共保存 {len(selected_indices)} 张图片。")
    return cpdfpqa_json

def image_to_base64(image_path):
    """
    将图片文件转换为 Base64 编码的字符串
    :param image_path: 图片文件的路径
    :return: Base64 编码的字符串
    """
    with open(image_path, "rb") as image_file:
        # 读取二进制数据并进行 Base64 编码
        base64_data = base64.b64encode(image_file.read()).decode("utf-8")
    return base64_data

def get_query_by_img(image_path): 
    base64_images=[]
    base64_str = image_to_base64(image_path) 
    base64_images.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{base64_str}"}
    })

    response  = AIclient.chat.completions.create(
        model="qwen-vl-max-latest", 
        messages=[
        {"role":"system","content":[{"type": "text", "text": "Your job is to ask a question based on the document page image provided by the user."}]},
        {
            "role": "user",
            "content": base64_images + [{"type": "text", "text": "Generate a question as the query statement for the retriever based on the document page image provided by the user, and meet the following requirements:\n 1. This query statement can only obtain answers through this document page. \n2. This query statement can be answered through single choice reasoning. \n3. This query statement cannot clearly display certain vocabulary on the current page, and semantic matching is required to retrieve this page.\n 4. To increase the difficulty, it is possible to add some other interfering vocabulary that does not change the semantics, but the sentence needs to be smooth. \nYour query statement:"}]
        }
        ]
    )

    return response.choices[0].message.content


def main():
    outputdir = "pdfs"

    orgin_arvix="c:/Users/dzy\Desktop/archive/arxiv-metadata-oai-snapshot.json"

    data = random_sample_jsonl(orgin_arvix, k=30)

    cpdfpqa_json = {"samples":[]}

    print(f"共随机抽取 {len(data)} 条记录:")
    for item in data:
        download_arxiv_pdf(item,outputdir)

    for item in data:
        pdfId = item["id"]
        if "/" in pdfId:
            paper_name = pdfId.replace('/', '_')
        else:
            paper_name = pdfId
        cpdfpqa_json = pdf_random_pages_to_png(f"pdfs/{paper_name}.pdf",cpdfpqa_json,10)
        
    with open(outputdir+"/cqdf_pqa.json", 'w', encoding='utf-8') as file2:
            json.dump(cpdfpqa_json, file2, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()

 