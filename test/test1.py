import os
import fitz  # PyMuPDF
from tkinter import Tk, filedialog, simpledialog, messagebox
import shutil


def pdf_to_images(pdf_path, output_folder, delete_pages=None):
    """
    将PDF每页转为图片，并跳过指定删除的页
    :param pdf_path: PDF文件路径
    :param output_folder: 输出文件夹
    :param delete_pages: 要删除的页码列表（从1开始）
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 打开PDF文件
    pdf_document = fitz.open(pdf_path)
    total_pages = pdf_document.page_count

    # 处理删除页参数
    if delete_pages is None:
        delete_pages = []
    else:
        # 转换为0-based索引并过滤无效页码
        delete_pages = [p - 1 for p in delete_pages if 1 <= p <= total_pages]

    saved_count = 0

    for page_num in range(total_pages):
        if page_num in delete_pages:
            continue  # 跳过删除页

        # 获取页面
        page = pdf_document[page_num]

        # 设置缩放因子 (提高DPI)
        zoom = 4  # 288 DPI (72*4)
        mat = fitz.Matrix(zoom, zoom)

        # 转换为图片 (RGB格式)
        pix = page.get_pixmap(matrix=mat, colorspace="rgb")

        # 生成输出路径
        output_path = os.path.join(output_folder, f"page_{page_num + 1}.png")
        pix.save(output_path)
        saved_count += 1

    pdf_document.close()
    return saved_count


def select_pdf_and_folder():
    """GUI交互选择文件和文件夹"""
    root = Tk()
    root.withdraw()  # 隐藏主窗口

    # 选择PDF文件
    pdf_path = filedialog.askopenfilename(
        title="选择PDF文件",
        filetypes=[("PDF文件", "*.pdf"), ("所有文件", "*.*")]
    )
    if not pdf_path:
        return

    # 选择输出文件夹
    output_folder = filedialog.askdirectory(title="选择输出文件夹")
    if not output_folder:
        return

    # 输入要删除的页码
    delete_input = simpledialog.askstring(
        "删除页面",
        "输入要删除的页码（用逗号分隔，例如：2,5,7）\n留空则保留所有页："
    )

    # 处理页码输入
    delete_pages = []
    if delete_input and delete_input.strip():
        try:
            delete_pages = [int(p.strip()) for p in delete_input.split(",")]
        except ValueError:
            messagebox.showerror("错误", "页码格式无效！请使用数字和逗号")
            return

    # 执行转换
    saved_count = pdf_to_images(pdf_path, output_folder, delete_pages)

    # 打开输出文件夹
    if saved_count > 0:
        messagebox.showinfo("完成", f"成功转换 {saved_count} 页图片！")
        os.startfile(output_folder)  # 在文件资源管理器中打开
    else:
        messagebox.showwarning("无操作", "未生成任何图片！")


if __name__ == "__main__":
    # 安装提示
    try:
        import fitz
    except ImportError:
        print("正在安装依赖库...")
        os.system("pip install PyMuPDF tkinter")

    select_pdf_and_folder()