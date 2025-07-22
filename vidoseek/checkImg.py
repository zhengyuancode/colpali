import os
from PyPDF2 import PdfReader

# 定义目录路径
PDF_DIR = "./pdf"
MINERU_DIR = "./minerU_pdf"

# 遍历 PDF 文件夹中的所有文件
for pdf_filename in os.listdir(PDF_DIR):
    if not pdf_filename.lower().endswith(".pdf"):
        continue

    # 提取 PDF 名称（不含扩展名）
    pdf_name = os.path.splitext(pdf_filename)[0]
    pdf_path = os.path.join(PDF_DIR, pdf_filename)

    # 获取 PDF 总页数
    try:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            page_count = len(reader.pages)
    except Exception as e:
        print(f"无法读取 PDF 文件 {pdf_name}: {e}")
        continue

    # 构造图片目录路径并统计 PNG 数量
    image_dir = os.path.join(MINERU_DIR, pdf_name+"/pages")
    png_count = 0

    if os.path.exists(image_dir) and os.path.isdir(image_dir):
        png_count = len(
            [f for f in os.listdir(image_dir) if f.lower().endswith(".png")]
        )

    # 比较页数与图片数量
    if page_count != png_count:
        print(pdf_name)
        
    if page_count == png_count:
        print(f"{pdf_name}的page_count:{page_count}与转换的页数{png_count}相同，转换充分")