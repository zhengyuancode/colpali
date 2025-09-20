import os
import logging
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json
from caption import getTextList,getTextList_local
from openai import OpenAI

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("uvicorn.access").propagate = True
logging.getLogger("uvicorn.error").propagate = True
import asyncio
from mineru_process import run_mineru

QWENAPIKEY="sk-f78b07615c8a45128d760579e6d42e1f"
AIclient = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=QWENAPIKEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


# 将给定路径下的图片样本转为pdf形式存储，为后续mineru解析做准备
def images_to_pdfs(image_folder,output_pdf_name):
    folder = Path(image_folder)
    
    # 检查路径是否存在且是目录
    if not folder.exists():
        print(f"错误：路径 '{folder}' 不存在。")
        return
    
    if not folder.is_dir():
        print(f"错误：'{folder}' 不是一个有效的文件夹。")
        return

    pdf_folder = folder / "pdf"
    pdf_folder.mkdir(exist_ok=True)

    # 支持的图片格式
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}
    
    # 获取所有图片文件并按名称排序（确保顺序一致）
    image_files = [f for f in folder.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        print("错误：未在指定文件夹中找到支持的图片文件。")
        return

    # 排序（按文件名字母序，可根据需要改为时间等）
    image_files.sort(key=lambda x: x.name)

    # 存储图像对象（用于转换为 PDF）
    images_for_pdf = []
    page_to_image_map = {}  # 记录：PDF页码 -> 图片文件名

    try:
        for idx, file_path in enumerate(image_files):
            image = Image.open(file_path)
            
            # 如果是 RGBA 或 LA，转为 RGB（PDF 不支持透明通道）
            if image.mode in ('RGBA', 'LA'):
                bg = Image.new('RGB', image.size, (255, 255, 255))
                bg.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                image = bg
            elif image.mode != 'RGB':
                image = image.convert('RGB')

            images_for_pdf.append(image)
            # 记录第 idx+1 页对应哪个图片
            page_to_image_map[idx + 1] = file_path.name

        # 合并为单个 PDF
        if images_for_pdf:
            pdf_path = pdf_folder / f"{output_pdf_name}.pdf"
            # 第一页作为基础
            images_for_pdf[0].save(
                pdf_path,
                "PDF",
                resolution=100.0,
                save_all=True,
                append_images=images_for_pdf[1:]  # 剩余页面
            )

            # 保存映射关系到 JSON 文件
            json_path = pdf_folder / "page_to_image.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(page_to_image_map, f, indent=2, ensure_ascii=False)
            print(f"映射文件已保存：{json_path}")

        # 关闭所有图像
        for img in images_for_pdf:
            img.close()

    except Exception as e:
        print(f"转换过程中发生错误：{e}")

async def convert_pdf(
    documentsPath, 
    datasetName
):    
    output_dir = documentsPath+"/parse"
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(documentsPath+"/pdf"):
        print("原始图像无法应用mineru解析，转换为pdf中")
        images_to_pdfs(documentsPath,"vidore_docvqa")
    
    pdf_path = documentsPath+f"/pdf/{datasetName}.pdf"
    # pdfID = Path(pdf_path).stem
    
    if not os.path.exists(output_dir):
        logger.info("开始执行minerU解析")
        await asyncio.to_thread(run_mineru, pdf_path, output_dir)
        logger.info("执行minerU解析结束")
    
    if not os.path.exists(str(output_dir)+f"/{datasetName}/caption_text_list.json"):
        logger.info("开始执行文档caption")
        caption_text_list_path = getTextList(
                                    str(output_dir)+f"/{datasetName}/auto/{datasetName}_content_list.json",
                                    "english",
                                    str(output_dir)+f"/{datasetName}/auto/",
                                    str(output_dir)+f"/{datasetName}/caption_text_list.json",
                                    AIclient
                                    )
        print(f"caption_text_list_path:{caption_text_list_path}")



def main():
    documentsPath = "/root/autodl-tmp/cpdfr-data/cpdfr/hybridSearch/experment_vidore/vidore_data/docvqa_test_subsampled/documents"
    datasetName = "vidore_docvqa"
    asyncio.run(convert_pdf(documentsPath,datasetName))
    
    
if __name__ == "__main__":
    main()