from pipeline import model_builder
import torch
import pandas as pd

from pipeline import const
CLASSES = const.CLASSES
class_to_idx = {cls_name: i for i, cls_name in enumerate(CLASSES)}
idx_to_class = {v: k for k, v in class_to_idx.items()}

def predict_testset(model: torch.nn.Module,
                    submission_number: int,
                    submission_dir: str,
                    test_dataloader: torch.utils.data.DataLoader,
                    device: torch.device):
    submission_data = []
    
    model.eval()
    with torch.inference_mode():
        for imgs, image_ids in test_dataloader:
            outputs = model(imgs.to(device))
            preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
    
            for img_id, pred in zip(image_ids, preds):
                predicted_label = idx_to_class[pred.item()]
                submission_data.append({'id': int(img_id), 'label': predicted_label})
                if(int(img_id) % 10000 == 0):
                    print(f"Done ảnh {img_id}")
                    
    # Convert the submission data to a DataFrame
    submission_df = pd.DataFrame(submission_data)
    
    # Save the submission file
    submission_df.to_csv(f'{submission_dir}/submission{submission_number}.csv', index=False)
    
    print(f"Submission file saved to 'submission{submission_number}.csv'.")