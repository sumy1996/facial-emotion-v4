import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
import cv2
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification
import threading
import numpy as np


# 载入本地模型
model_str = "./model"  # 本地模型路径
model = ViTForImageClassification.from_pretrained(model_str, num_labels=7)
labels_list = ['sad', 'disgust', 'angry', 'neutral', 'fear', 'surprise', 'happy']


class EmotionAnalyzerApp:
    def __init__(self, master):
        self.master = master
        master.title("Emotion Analyzer")
        master.geometry('300x350')  # 调整窗口大小

        # 使用 ttk 改进按钮和标签的外观
        self.frame = ttk.Frame(master)
        self.frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # 按钮：选择图片文件
        self.btn_select_images = ttk.Button(self.frame, text="选择图片文件", command=self.select_images)
        self.btn_select_images.pack(fill=tk.X, pady=5)

        # 按钮：分析图片文件
        self.btn_start_images_analysis = ttk.Button(self.frame, text="开始图片分析", command=self.start_images_analysis_thread)
        self.btn_start_images_analysis.pack(fill=tk.X, pady=5)

        # 按钮：选择视频文件
        self.btn_select_video = ttk.Button(self.frame, text="选择视频文件", command=self.select_video)
        self.btn_select_video.pack(fill=tk.X, pady=5)

        # 按钮：设置抽帧频率
        self.btn_frame_rate = ttk.Button(self.frame, text="设置抽帧频率", command=self.set_frame_rate)
        self.btn_frame_rate.pack(fill=tk.X, pady=5)

        # 按钮：选择保存路径
        self.btn_save_path = ttk.Button(self.frame, text="选择保存路径", command=self.select_save_path)
        self.btn_save_path.pack(fill=tk.X, pady=5)

        # 按钮：开始分析
        self.btn_start_analysis = ttk.Button(self.frame, text="开始分析", command=self.start_analysis_thread)
        self.btn_start_analysis.pack(fill=tk.X, pady=5)

        # 按钮：取消操作
        self.btn_cancel = ttk.Button(self.frame, text="取消操作", command=self.cancel_analysis)
        self.btn_cancel.pack(fill=tk.X, pady=5)
        self.btn_cancel['state'] = 'disabled'  # 初始时禁用取消按钮

        # 进度条
        self.progress = ttk.Progressbar(self.frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.progress.pack(pady=10)

        self.image_save_path = None
        self.video_path = None
        self.save_path = None
        self.frame_rate = 30
        self.running = False  # 控制分析过程的变量

    def select_images(self):
        filetypes = [('JPG files', '*.jpg'), ('JPEG files', '*.jpeg')]
        self.image_paths = filedialog.askopenfilenames(title="选择图片文件", filetypes=filetypes)
        if self.image_paths:
            print("选择的图片路径:", self.image_paths)
        self.image_save_path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                                            filetypes=[("Excel files", "*.xlsx")])
        if self.image_save_path:
            print("图片保存路径:", self.image_save_path)

    def start_images_analysis_thread(self):
        if not self.image_paths or not self.image_save_path:
            messagebox.showerror("错误", "请确保已选择图片文件和图片保存路径")
            return
        self.running = True  # 如果你需要使用运行状态来控制线程，可以这样设置
        self.btn_start_images_analysis['state'] = 'disabled'  # 禁用按钮，防止重复点击
        self.btn_cancel['state'] = 'normal'  # 启用取消按钮
        threading.Thread(target=self.start_images_analysis).start()

    def start_images_analysis(self):
        results = process_images(self.image_paths, self.progress)
        if results is not None:
            save_results_to_excel(results, self.image_save_path)
            messagebox.showinfo("完成", "图片处理完成，结果已保存")
        self.progress['value'] = 0

    def select_video(self):
        self.video_path = filedialog.askopenfilename(title="选择视频文件", filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if self.video_path:
            print("选择的视频路径:", self.video_path)

    def set_frame_rate(self):
        self.frame_rate = simpledialog.askinteger("抽帧频率", "请输入抽帧频率（默认为30）:", minvalue=1, initialvalue=30)

    def select_save_path(self):
        self.save_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
        if self.save_path:
            print("保存路径:", self.save_path)

    def start_analysis_thread(self):
        if not self.video_path or not self.save_path:
            messagebox.showerror("错误", "请确保已选择视频文件和保存路径")
            return
        self.running = True
        self.btn_start_analysis['state'] = 'disabled'
        self.btn_cancel['state'] = 'normal'
        threading.Thread(target=self.start_analysis).start()

    def start_analysis(self):
        results = process_video(self.video_path, self.frame_rate, self.progress, self.running)
        if results is not None:
            save_results_to_excel(results, self.save_path)
            messagebox.showinfo("完成", "处理完成，结果已保存")
        self.progress['value'] = 0
        self.btn_start_analysis['state'] = 'normal'
        self.btn_cancel['state'] = 'disabled'

    def cancel_analysis(self):
        self.running = False  # 控制图片分析和视频分析的通用变量
        self.btn_cancel['state'] = 'disabled'  # 禁用取消按钮
        self.btn_start_images_analysis['state'] = 'normal'  # 重新启用开始图片分析按钮
        self.btn_start_analysis['state'] = 'normal'  # 重新启用开始视频分析按钮
        messagebox.showinfo("已取消", "分析已取消")

def process_images(image_paths, progress):
    results = []
    total_images = len(image_paths)
    for i, path in enumerate(image_paths):
        image = Image.open(path)
        emotion_probs = predict(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
        results.append((path, emotion_probs))
        progress['value'] = ((i + 1) / total_images) * 100
        progress.update()
    return results

def transform_image(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小到224x224
        transforms.ToTensor(),  # 将图像转换为Tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化
    ])
    return transform(image).unsqueeze(0)  # 添加批次维度

def predict(frame):
    tensor_image = transform_image(frame)
    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():  # 关闭梯度计算
        outputs = model(tensor_image)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return predictions.numpy().flatten()  # 返回概率数组

def process_video(video_path, frame_rate, progress, running):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    results = []
    for i in range(0, frame_count, frame_rate):
        if not running:
            print("分析被取消")
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        emotion_probs = predict(frame)
        results.append((i, emotion_probs))
        progress['value'] = (i / frame_count) * 100
        progress.update()
    cap.release()
    return results


def save_results_to_excel(results, file_name):
    emotions = ['sad', 'disgust', 'angry', 'neutral', 'fear', 'surprise', 'happy']
    formatted_results = []

    # Check if results are from images or video frames based on the tuple length
    if all(len(result) == 2 for result in results):
        key = 'Image' if isinstance(results[0][0], str) else 'Frame'
        for item, emotion_probs in results:
            result = {key: item}
            for index, emo in enumerate(emotions):
                result[emo] = emotion_probs[index]
            formatted_results.append(result)
    else:
        print("Error: Unexpected data structure in results")
        return

    # Create a DataFrame from formatted results
    columns_order = [key] + emotions
    df = pd.DataFrame(formatted_results)
    df = df[columns_order]  # Ensure the columns are in the correct order

    # Save the DataFrame to Excel
    try:
        df.to_excel(file_name, index=False)
        print("文件保存成功，路径为:", file_name)
    except Exception as e:
        print("保存文件失败:", e)


def main():
    root = tk.Tk()
    app = EmotionAnalyzerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
