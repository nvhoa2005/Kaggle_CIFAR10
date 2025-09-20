# Kaggle_CIFAR10
# Link: https://www.kaggle.com/competitions/cifar-10

Notebook này huấn luyện và suy luận phân loại ảnh **CIFAR-10** trên Kaggle bằng **PyTorch** với mục đích làm quen với Pytorch. Dự án gồm 3 mô hình: một CNN nhỏ **TinyVGG** (train from scratch) và hai phiên bản **EfficientNet-B0** dùng transfer learning, phiên bản đầu tiên mở hai layer cuối, phiên bản thứ hai mở toàn bộ layer. Cuối cùng tạo file `submission.csv` theo định dạng Kaggle.

-------------------------------

## Dataset
- Competition: **Kaggle CIFAR-10** (10 lớp: `airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`).
- Cấu trúc dữ liệu trong notebook:
  - `train_dir = /kaggle/working/train/train` (ảnh `.png`)
  - `test_dir  = /kaggle/working/test/test` (ảnh `.png`)
  - Nhãn train đọc từ: `/kaggle/input/cifar-10/trainLabels.csv`
  - Mẫu submission: `/kaggle/input/cifar-10/sampleSubmission.csv`

## Thiết bị: Kaggle Notebook với **GPU T4**, PyTorch + Torchvision.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
---> agnostic code

### Transforms
- Giai đoạn TinyVGG (ảnh 32×32): dùng
  - `RandomHorizontalFlip(p=0.5)`
  - `ToTensor()`

- Giai đoạn EfficientNet-B0 (để tận dụng pretrain ImageNet): **Resize 224×224 + Normalize ImageNet**
```
manual_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

### Dataset tùy biến
- **Train**: `ImageFolderCustom` đọc ảnh từ thư mục và map nhãn từ `trainLabels.csv`.
- **Test**: `TestDataset` trả về `(img, image_id)` để batch inference.

### DataLoader
```
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
```

---

## Mô hình

### 1 TinyVGG (train from scratch)
- Kiến trúc toy CNN: 2 conv-block (Conv→ReLU→Conv→ReLU→MaxPool) + Flatten + Linear.
- Dùng cho ảnh kích thước nhỏ (CIFAR-10).

### 2 EfficientNet-B0 — **partial unfreeze**
- **Freeze toàn bộ backbone**, **mở** các **layer cuối** `features[8]` + `classifier`:
```
    for param in model_2.features.parameters():
        param.requires_grad = False
    for param in model_2.features[8].parameters():   # Conv2dNormActivation
        param.requires_grad = True
    for param in model_2.classifier.parameters():    # classifier
        param.requires_grad = True
```
- Thay **classifier**: `Dropout(0.2) → Linear(1280 → num_classes)`.

### 3 EfficientNet-B0 — **full fine-tuning**
- Tương tự trên nhưng **mở trainable cho toàn bộ `features`**:
```
    for param in model_2.features.parameters():
        param.requires_grad = False
```

---

## Huấn luyện

Các hàm chính:
- `train_step()` và `train()` (ghi lại `train_loss`, `train_acc` theo epoch).
- Loss: `nn.CrossEntropyLoss()`
- Optimizer: `torch.optim.Adam(lr=1e-3)`
- Số epoch mặc định: **5**

Nhớ đặt `model.train()` khi huấn luyện, `model.eval()` + `torch.inference_mode()` khi suy luận.

### Kết quả (log từ notebook)
- **EfficientNet-B0 (partial unfreeze)** — 5 epoch  
    `train_acc` tăng từ **0.7763 → 0.8835**, tổng thời gian ~ **435s**.
- **EfficientNet-B0 (full fine-tuning)** — 5 epoch  
    `train_acc` tăng từ **0.8495 → 0.9483**, tổng thời gian ~ **1320s**.

Lưu ý: notebook chỉ log **train_acc**, bạn có thể tách **validation set** để theo dõi overfit.

---

## Suy luận & Tạo Submission

### Cách 1 (đơn chiếc – chậm)
- Lặp qua từng `id` trong `sampleSubmission.csv`, load ảnh, transform, `model(img)`.

### Cách 2 (khuyến nghị – nhanh)
- Dùng `TestDataset` + `DataLoader(batch_size=32)` để **batch inference trên GPU**:

**Định dạng bắt buộc của Kaggle**:
id,label
1,frog
2,cat
...

## Cấu trúc/Ý nghĩa các cell chính trong Notebook
- Chuẩn bị đường dẫn, đọc CSV nhãn, đếm số ảnh.
- Viết **Dataset** tùy biến cho train/test.
- Tạo **DataLoader** (batch size 32).
- Định nghĩa **TinyVGG** và **EfficientNet-B0** (2 biến thể).
- Vòng lặp **train()** với Adam + CrossEntropy, 5 epoch.
- Hai cách **inference** và ghi vào `submission.csv`.

## Kết quả
- Model1 - TinyVGG: accuracy = 0.55970
- Model2 - EfficientNet-B0 (trainable 2 layer cuối): accuracy = 0.86900
- Model3 - EfficientNet-B0 (trainable toàn bộ layer): accuracy = 0.93350 (Highest)

