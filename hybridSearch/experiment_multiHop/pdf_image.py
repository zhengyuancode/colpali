import fitz  # pip install pymupdf
from tqdm import tqdm
import os
import time


def pdfToImage(pdf_path,output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # 打开PDF文件
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    # 设置DPI（200 DPI）
    dpi = 200
    zoom = dpi / 72  # fitz默认72DPI

    with tqdm(total=total_pages, desc="Convert", unit="page") as pbar:
        for i in range(total_pages):
            page = doc.load_page(i)
            
            # 创建高质量图像
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            
            # 保存图像
            output_path = os.path.join(output_dir, f"{i+1}.png")
            pix.save(output_path)
            
            # 释放内存
            del page, pix
            pbar.update(1)

    doc.close()

