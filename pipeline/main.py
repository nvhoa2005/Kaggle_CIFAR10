import os
import py7zr
import torch

def main():
    # 1. Cài thư mục cần thiết
    os.makedirs("pipeline", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("submissions", exist_ok=True)
    
    # 2. Đường dẫn đến file .7z
    train_file = '/kaggle/input/cifar-10/train.7z'
    test_file  = '/kaggle/input/cifar-10/test.7z'
    
    train_output_dir = '/kaggle/working/data/train/'
    test_output_dir  = '/kaggle/working/data/test/'
    
    # 3. Giải nén dữ liệu train
    with py7zr.SevenZipFile(train_file, mode='r') as z:
        z.extractall(path=train_output_dir)
    print("[INFO] Giải nén thành công train.7z")
    
    # 4. Giải nén dữ liệu test
    with py7zr.SevenZipFile(test_file, mode='r') as z:
        z.extractall(path=test_output_dir)
    print("[INFO] Giải nén thành công test.7z")
    
    print("[INFO] DONE extract!")
    
    # 5. Gọi script train.py trong pipeline
    print("[INFO] Bắt đầu train...")
    os.system("python -m pipeline.train")
if __name__ == "__main__":
    main()