import argparse
import json
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
import torch
from torchvision.models import resnet50
from tqdm import tqdm

import loss
from dataset import download_dataset


class TestResulutJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, TestResult):
            return {'loss': obj.loss.tolist(), 
                    'retino_confusion_matrix': obj.retino_conf_mat.data.tolist(),
                    'edema_confusion_matrix': obj.edema_conf_mat.data.tolist(),
                    }


class ConfusionMatrix():
    def __init__(self, data : NDArray[np.int32]):
        self.data = data
    
    @property
    def accuracy(self):
        return np.sum(np.diag(self.data)) / np.sum(self.data)
    
    @property
    def recall(self):        
        return np.diag(self.data) / np.sum(self.data, axis=1)    
    
    @property
    def precision(self):
        return np.diag(self.data) / (np.sum(self.data, axis=0) + 1e-8)
    
    @property
    def f1_score(self):
        return 2 * self.recall * self.precision / (self.recall + self.precision + 1e-8)
    
    @property
    def macro_avg_recall(self):
        return np.mean(self.recall)
    
    @property
    def macro_avg_precision(self):
        return np.mean(self.precision)
    
    @property
    def macro_avg_f1_score(self):
        return np.mean(self.f1_score)


@dataclass
class TestResult:
    loss: NDArray[np.float32]
    retino_conf_mat: ConfusionMatrix
    edema_conf_mat: ConfusionMatrix


def consume_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size', type=int, default=2, dest='batch_size', help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, dest='num_workers', help='Number of workers for dataloader')
    parser.add_argument('--model-path', type=str, dest='model_path', help='Path to the model.')
    parser.add_argument('--gamma', type=float, default=2, dest='gamma', help='Gamma for Focal Loss')
    
    return parser.parse_args()


def load_model(model_path, model):
    data = torch.load(model_path, weights_only=False)
    if 'model' in data:
        model.load_state_dict(data['model'])
    else:
        model.load_state_dict(data)


def test(model, loader, loss_fn, device=torch.get_default_device) -> TestResult:

    test_result = TestResult(np.zeros((len(loss_fn),), dtype=np.float32),
                             ConfusionMatrix(np.zeros((5, 5), dtype=np.int32)),
                             ConfusionMatrix(np.zeros((2, 2), dtype=np.int32)))
    
    if not isinstance(loss_fn, list):
        loss_fn = [loss_fn]
    
    model.eval()
    
    with tqdm(total=len(loader)) as pbar:
        with torch.no_grad():
            for (x, y) in loader:
                x, y = x.to(device), y.to(device)
                
                pred = model(x)
                
                # Accumulate losses
                for i in range(len(loss_fn)):
                    test_result.loss[i] += loss_fn[i](pred, y).item()
                
                # Count correct predictions
                for i in range(len(pred)):
                    test_result.retino_conf_mat.data[y[i,0].item(), pred[i,:5].argmax(dim=0).item()] += 1
                    
                # Count correct predictions
                for i in range(len(pred)):
                    test_result.edema_conf_mat.data[y[i,1], pred[i,5].argmax(dim=0).item()] += 1
                
                pbar.set_description('Validation Progress:')
                pbar.update(1)
    
    
        #Calculate losses
        for i in range(len(loss_fn)):
            test_result.loss[i] /= len(loader)
    
    return test_result


if __name__ == '__main__':
    print('Starting...')
    
    
    # Set Device
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    
    
    # Consume arguments
    args = consume_args()  


    # Downlaod Dataset
    print('Download Dataset...')
    train_loader, test_loader, val_loader = download_dataset(args.batch_size, args.num_workers)


    # Load Class Distributions
    test_retino_class_dist, test_edema_class_dist = test_loader.dataset.class_dist()
    test_retino_class_dist = torch.tensor(test_retino_class_dist, dtype=torch.float32, device=device)
    test_edema_class_dist = torch.tensor(test_edema_class_dist, dtype=torch.float32, device=device)
    

    # Define Loss Functions
    focal_loss_fn = { 
                    'test': loss.FocalRetinalDiseaseLoss(gamma=args.gamma, alpha_retino=test_retino_class_dist, alpha_edema=test_edema_class_dist[0].item()),
                    }
    
    
    normal_loss_fn = loss.NormalRetinalDiseaseLoss()
    
    
    # Define Model
    model = resnet50().to(device=device)
    
    # Load Checkpoint
    load_model(args.model_path, model)
        
    test_loss_fns = [normal_loss_fn, focal_loss_fn['test']]
    test_result = test(model, val_loader, test_loss_fns, device)
    
    print(f"Test Loss (Normal): {test_result.loss[0]:.4f}")
    print(f"Test Loss (Focal): {test_result.loss[1]:.4f}")
        
    print("\n")
        
    print("Retinopathy Grade:")
    print(f"Accuracy: {test_result.retino_conf_mat.accuracy:.3f}")
    print("Recall per class:")
    for i in range(5):
        print(f"\t Grade {i}: {test_result.retino_conf_mat.recall[i]:.3f}")
    print(f"Macro Avg Recall: {test_result.retino_conf_mat.macro_avg_recall:.3f}")
    print("Precision per class:")
    for i in range(5):
        print(f"\t Grade {i}: {test_result.retino_conf_mat.precision[i]:.3f}")
    print(f"Macro Avg Precision: {test_result.retino_conf_mat.macro_avg_precision:.3f}")
    print("F1 Score per class:")
    for i in range(5):
        print(f"\t Grade {i}: {test_result.retino_conf_mat.f1_score[i]:.3f}")
    print(f"Macro Avg F1 Score: {test_result.retino_conf_mat.macro_avg_f1_score:.3f}")
        
    print("\n")
    
    print("Risk of Edema:")
    print(f"Accuracy: {test_result.edema_conf_mat.accuracy:.3f}")
    print("Recall per class:")
    for i in range(2):
        print(f"\t Risk {bool(i)}: {test_result.edema_conf_mat.recall[i]:.3f}")
    print(f"Macro Avg Recall: {test_result.edema_conf_mat.macro_avg_recall:.3f}")
    print("Precision per class:")
    for i in range(2):
        print(f"\t Risk {bool(i)}: {test_result.edema_conf_mat.precision[i]:.3f}")
    print(f"Macro Avg Precision: {test_result.edema_conf_mat.macro_avg_precision:.3f}")
    print("F1 Score per class:")
    for i in range(2):
        print(f"\t Risk {bool(i)}: {test_result.edema_conf_mat.f1_score[i]:.3f}")
    print(f"Macro Avg F1 Score: {test_result.edema_conf_mat.macro_avg_f1_score:.3f}")
    print("-------------------------------")
    print("Done!")
    
    