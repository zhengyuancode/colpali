import fitz  # pip install pymupdf
from tqdm import tqdm
import os
import time

pdf_path = "./pdf"
output_dir = "./minerU_pdf/"

def pdfToImage(pdf_path,output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print("开始转换并写入图片...")
    start_time = time.time()

    # 打开PDF文件
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"PDF加载完成，共 {total_pages} 页，开始转换...")

    # 设置DPI（200 DPI）
    dpi = 200
    zoom = dpi / 72  # fitz默认72DPI

    with tqdm(total=total_pages, desc="转换进度", unit="页") as pbar:
        for i in range(total_pages):
            page = doc.load_page(i)
            
            # 创建高质量图像
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            
            # 保存图像
            output_path = os.path.join(output_dir, f"page_{i+1}.png")
            pix.save(output_path)
            
            # 释放内存
            del page, pix
            pbar.update(1)

    doc.close()
    print(f"转换完成！总耗时 {time.time()-start_time:.1f}秒")
    
def get_all_files(folder_path):
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            full_path = os.path.join(root, file)
            file_list.append(full_path)
    return file_list

def extract_pdf_names(path):
    return os.path.splitext(os.path.basename(path))[0]

def main():
    files = get_all_files(pdf_path)
    print(files)
    for path in files:
        pdfToImage(path,output_dir+extract_pdf_names(path)+"/pages")

if __name__ == "__main__":
    main()