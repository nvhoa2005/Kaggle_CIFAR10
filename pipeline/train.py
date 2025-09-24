import os
import torch
from pipeline import data_setup, engine, model_builder, utils, const, predict
import pandas as pd

from torchvision import transforms

import importlib
importlib.reload(data_setup)
importlib.reload(engine)
importlib.reload(model_builder)
importlib.reload(utils)
importlib.reload(const)
importlib.reload(predict)

# Setup hyperparameters
EPOCHS = const.EPOCHS
BATCH_SIZE = const.BATCH_SIZE
CLASSES = const.CLASSES
HIDDEN_UNITS = len(CLASSES)
LEARNING_RATE = const.LEARNING_RATE
NUM_WORKERS = const.NUM_WORKERS

# Setup directories
kaggle_path = "/kaggle/working"
data_dir = os.path.join(kaggle_path, "data")

# Đường dẫn thư mục
train_dir = os.path.join(data_dir, "train", "train")
test_dir = os.path.join(data_dir, "test", "test")

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# DataFrame
train_labels = pd.read_csv("/kaggle/input/cifar-10/trainLabels.csv")
sample_submissions = pd.read_csv("/kaggle/input/cifar-10/sampleSubmission.csv")

# Create transform
train_transform = transforms.Compose([
    transforms.Resize((224, 224)), # 1. Reshape all images to 224x224 (though some models may require different sizes)
    transforms.ToTensor(), # 2. Turn image values to between 0 & 1 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # 3. A mean of [0.485, 0.456, 0.406] (across each colour channel)
                         std=[0.229, 0.224, 0.225]) # 4. A standard deviation of [0.229, 0.224, 0.225] (across each colour channel),
])
test_transform = transforms.Compose([
    transforms.Resize((224, 224)), # 1. Reshape all images to 224x224 (though some models may require different sizes)
    transforms.ToTensor(), # 2. Turn image values to between 0 & 1 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # 3. A mean of [0.485, 0.456, 0.406] (across each colour channel)
                         std=[0.229, 0.224, 0.225]) # 4. A standard deviation of [0.229, 0.224, 0.225] (across each colour channel),
])

# Create train/test dataloader and get class names as a list
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                               test_dir=test_dir,
                                                                               train_transform=train_transform,
                                                                               test_transform=test_transform,
                                                                               train_labels=train_labels,
                                                                               test_labels=sample_submissions,
                                                                               batch_size=BATCH_SIZE,
                                                                               num_workers=NUM_WORKERS)

# Create model with help from model_builder.py
model = model_builder.EfficientNetB0(in_features=1280,
                                        pretrained = True,
                                          output_shape=len(CLASSES)).to(device)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# Start training with help from engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             optimizer=optimizer,
             loss_fn=loss_fn,
             epochs=EPOCHS,
             device=device)

# Save the model with help from utils.py
utils.save_model(model=model,
           target_dir=os.path.join(kaggle_path, "models"),
           model_name="example_model.pth")

loaded_model = model_builder.EfficientNetB0(
    in_features=1280,
    pretrained = True,
    output_shape=len(CLASSES)).to(device)         

# Load trọng số
model_path = os.path.join(kaggle_path, "models", "example_model.pth")
loaded_model.load_state_dict(torch.load(model_path, map_location=device))

# Predict và lưu vào file submission
submission_dir = os.path.join(kaggle_path, "submissions")

predict.predict_testset(model=loaded_model,
                        submission_number=1,
                        submission_dir=submission_dir,
                        test_dataloader=test_dataloader,
                        device=device)