import os
from datasets import Dataset
from PIL import Image
import io
import json
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def docvqa_preprocess():
    # 配置路径
    parquet_path = "/home/gpu/dzy/M3-CaseRAG/experiment_singleHop/vidore/docvqa/test-00000-of-00001.parquet"
    output_dir = "/home/gpu/dzy/M3-CaseRAG/experiment_singleHop/vidore/docvqa/pages"
    pdf_path = "/home/gpu/dzy/M3-CaseRAG/experiment_singleHop/vidore/docvqa/docvqa.pdf"
    json_output = "/home/gpu/dzy/M3-CaseRAG/experiment_singleHop/vidore/docvqa/docvqa.json"

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据集
    dataset = Dataset.from_parquet(parquet_path)

    # 存储所有图像（用于后续生成 PDF）
    images_for_pdf = []
    data = [] 
    page_id=[]

    # 遍历每一行
    for idx, example in enumerate(dataset):
        image_data = example["image"]
        if example["image_filename"]+"|"+str(example["page"]) in page_id:
            print(f"Duplicate page found: {example['image_filename']} page {example['page']}, error.")
            return
        else:
            page_id.append(example["image_filename"]+"|"+str(example["page"]))
        # 解析图像数据
        if isinstance(image_data, bytes):
            try:
                img = Image.open(io.BytesIO(image_data)).convert("RGB")
            except Exception as e:
                print(f"Error opening image at row {idx}: {e}")
                continue
        elif isinstance(image_data, str):
            import base64
            try:
                img_bytes = base64.b64decode(image_data)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception as e:
                print(f"Error decoding base64 image at row {idx}: {e}")
                continue
        elif hasattr(image_data, "convert"):
            img = image_data.convert("RGB")
        else:
            print(f"Unsupported image type at row {idx}: {type(image_data)}")
            continue

        # 保存单张 PNG
        png_path = os.path.join(output_dir, f"{idx + 1}.png")
        img.save(png_path, "PNG")
        print(f"Saved {png_path}")

        # 添加到 PDF 图像列表（必须是 RGB）
        images_for_pdf.append(img)
        data.append({
                "query": example["query"],
                "page_num": idx + 1
            })

    # 生成多页 PDF
    if images_for_pdf:
        # 第一页作为主图像，其余作为附加页
        first_img = images_for_pdf[0]
        extra_imgs = images_for_pdf[1:]
        first_img.save(pdf_path, "PDF", save_all=True, append_images=extra_imgs)
        print(f"PDF saved to: {pdf_path}")
    else:
        print("No valid images to create PDF.")
    with open(json_output, 'w') as f:
        json.dump(data, f, indent=4)
        
def tatdqa_preprocess():
    
    # 配置路径
    parquet_files = [
        "/home/gpu/dzy/M3-CaseRAG/experiment_singleHop/vidore/tatdqa/test-00000-of-00002.parquet",
        "/home/gpu/dzy/M3-CaseRAG/experiment_singleHop/vidore/tatdqa/test-00001-of-00002.parquet"
    ]
    output_dir = "/home/gpu/dzy/M3-CaseRAG/experiment_singleHop/vidore/tatdqa/pages"
    pdf_output = "/home/gpu/dzy/M3-CaseRAG/experiment_singleHop/vidore/tatdqa/tatdqa.pdf"
    json_output = "/home/gpu/dzy/M3-CaseRAG/experiment_singleHop/vidore/tatdqa/tatdqa.json"

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 初始化数据结构
    page_id_to_page_num = {}  # 记录 image_filename+page -> page_num
    data = []                  # 存储查询字典

    # 处理每个parquet文件
    for parquet_file in parquet_files:
        df = pd.read_parquet(parquet_file)
        for _, row in df.iterrows():
            # 提取关键字段
            image_filename = row['image_filename']
            page = str(row['page'])  # 确保page转为字符串
            query = row['query']
            
            # 构建唯一标识
            key = f"{image_filename}|{page}"  # 使用分隔符避免路径冲突
            
            # 处理新图片
            if key not in page_id_to_page_num:
                # 分配新的page_num (从1开始)
                page_num = len(page_id_to_page_num) + 1
                page_id_to_page_num[key] = page_num
                
                # 保存图片
                try:
                    image_data = row["image"]["bytes"]
                    # 解析图像数据
                    if isinstance(image_data, bytes):
                        try:
                            img = Image.open(io.BytesIO(image_data)).convert("RGB")
                        except Exception as e:
                            print(f"Error opening image at row {_}: {e}")
                            continue
                    elif isinstance(image_data, str):
                        import base64
                        try:
                            img_bytes = base64.b64decode(image_data)
                            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        except Exception as e:
                            print(f"Error decoding base64 image at row {_}: {e}")
                            continue
                    elif hasattr(image_data, "convert"):
                        img = image_data.convert("RGB")
                    else:
                        print(f"Unsupported image type at row {_}: {type(image_data)}")
                        continue
                    
                    img_path = os.path.join(output_dir, f"{page_num}.png")
                    img.save(img_path, "PNG")
                except Exception as e:
                    print(f"Error processing image {image_filename}: {str(e)}")
                    continue  # 跳过错误图片
            
            # 获取page_num (新旧都一样)
            page_num = page_id_to_page_num[key]
            
            # 添加到数据列表
            data.append({
                "query": query,
                "page_num": page_num
            })

    # 生成PDF
    print(f"Generating PDF with {len(page_id_to_page_num)} pages...")
    c = canvas.Canvas(pdf_output, pagesize=letter)
    width, height = letter

    for page_num in range(1, len(page_id_to_page_num) + 1):
        img_path = os.path.join(output_dir, f"{page_num}.png")
        if os.path.exists(img_path):
            c.drawImage(img_path, 0, 0, width, height)
            c.showPage()
        else:
            print(f"Warning: Image {img_path} not found, skipping page {page_num}")

    c.save()

    # 保存JSON
    print(f"Saving JSON to {json_output}")
    with open(json_output, 'w') as f:
        json.dump(data, f, indent=4)

    print("Processing complete!")
    print(f"PDF saved to: {pdf_output}")
    print(f"JSON saved to: {json_output}")
    print(f"Total unique pages: {len(page_id_to_page_num)}")

def main():
    docvqa_preprocess()
        
if __name__ == "__main__":
    main()