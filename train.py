import torch
import os
import argparse
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
import loss
from test import test
from tqdm import tqdm
from dataset import download_dataset
import json
from test import TestResulutJsonEncoder


def consume_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--gamma', type=float, default=2, dest='gamma', help='Gamma for Focal Loss')
    parser.add_argument('--output', type=str, default=r'experiments', dest='output', help='Dir to save the model')
    parser.add_argument('--save-inv', type=int, default=1, dest='save_inv', help='Number of epochs to save the model')
    parser.add_argument('-sp', '--starting-point', type=str, default=None, dest='starting_point', help='checkpoint to start from')
    parser.add_argument('-e', '--epochs', type=int, default=10, dest='epochs', help='Number of epochs to train for')
    parser.add_argument('-b', '--batch-size', type=int, default=4, dest='batch_size', help='Batch size')
    parser.add_argument('--step-inv', type=int, default=1, dest='step_inv', help='Number of steps to perform an optimizer step')    
    parser.add_argument('--num-workers', type=int, default=4, dest='num_workers', help='Number of workers for dataloader')
    parser.add_argument('--batch-print-inv', type=int, default=1, dest='batch_print_inv', help='Number of batches to print after')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-5, dest='learning_rate', help='Learning rate')
    parser.add_argument('-wd', '--weight-decay', type=float, default=0, dest='weight_decay', help='Weight decay')
    parser.add_argument('-m', '--momentum', type=float, default=0.9, dest='momentum', help='Momentum')

    return parser.parse_args()


def train(model : torch.nn.Module, 
          loader : torch.utils.data.DataLoader,
          optimizer : torch.optim.Optimizer, 
          loss_fn : torch.nn.Module, 
          step_inv : int, 
          print_inv : int, 
          device : torch.device) -> float:
    
    model.train()
    total_loss_sum = 0.0
    with tqdm(total=len(loader)) as pbar:
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            
            pred = model(x)
            loss = loss_fn(pred, y)
            
            total_loss_sum += loss.item()
            
            loss.backward()
            
            if batch_idx % step_inv == 0 or batch_idx == len(loader) - 1:
                optimizer.step()    
                optimizer.zero_grad()
            
            if batch_idx % print_inv == 0:
                total_loss = total_loss_sum / (batch_idx + 1)
                pbar.set_description('Total Loss: {:.6f}'.format(
                        total_loss)
                        )
            
            pbar.update(1)
    
    return total_loss_sum / len(loader)


def load_model(model_path, model, optimizer):
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch'] + 1


def load_stats(stats_path):
    stats = []
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            stats = json.load(f)
    return stats


def save_checkpoint(model_path, model, optimizer, e, train_loss, val_loss):
    print("Saving Checkpoint...")
    checkpoint = {
                'epoch': e,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

    # Save the checkpoint
    torch.save(checkpoint, model_path)


def print_test_result(val_result):
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Test Loss (Normal): {val_result.loss[0]:.4f}")
    print(f"Test Loss (Focal): {val_result.loss[1]:.4f}")
        
    print("\n")
        
    print("Retinopathy Grade:")
    print(f"Accuracy: {val_result.retino_conf_mat.accuracy:.3f}")
    print("Recall per class:")
    for i in range(5):
        print(f"\t Grade {i}: {val_result.retino_conf_mat.recall[i]:.3f}")
    print(f"Macro Avg Recall: {val_result.retino_conf_mat.macro_avg_recall:.3f}")
    print("Precision per class:")
    for i in range(5):
        print(f"\t Grade {i}: {val_result.retino_conf_mat.precision[i]:.3f}")
    print(f"Macro Avg Precision: {val_result.retino_conf_mat.macro_avg_precision:.3f}")
    print("F1 Score per class:")
    for i in range(5):
        print(f"\t Grade {i}: {val_result.retino_conf_mat.f1_score[i]:.3f}")
    print(f"Macro Avg F1 Score: {val_result.retino_conf_mat.macro_avg_f1_score:.3f}")
        
    print("\n")
    
    print("Risk of Edema:")
    print(f"Accuracy: {val_result.edema_conf_mat.accuracy:.3f}")
    print("Recall per class:")
    for i in range(2):
        print(f"\t Risk {bool(i)}: {val_result.edema_conf_mat.recall[i]:.3f}")
    print(f"Macro Avg Recall: {val_result.edema_conf_mat.macro_avg_recall:.3f}")
    print("Precision per class:")
    for i in range(2):
        print(f"\t Risk {bool(i)}: {val_result.edema_conf_mat.precision[i]:.3f}")
    print(f"Macro Avg Precision: {val_result.edema_conf_mat.macro_avg_precision:.3f}")
    print("F1 Score per class:")
    for i in range(2):
        print(f"\t Risk {bool(i)}: {val_result.edema_conf_mat.f1_score[i]:.3f}")
    print(f"Macro Avg F1 Score: {val_result.edema_conf_mat.macro_avg_f1_score:.3f}")
    print("-------------------------------")


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
    print('Load Class Distributions...')
    train_retino_class_dist, train_edema_class_dist = train_loader.dataset.class_dist()
    train_retino_class_dist = torch.tensor(train_retino_class_dist, dtype=torch.float32, device=device)
    train_edema_class_dist = torch.tensor(train_edema_class_dist, dtype=torch.float32, device=device)
    test_retino_class_dist, test_edema_class_dist = test_loader.dataset.class_dist()
    test_retino_class_dist = torch.tensor(test_retino_class_dist, dtype=torch.float32, device=device)
    test_edema_class_dist = torch.tensor(test_edema_class_dist, dtype=torch.float32, device=device)
    val_retino_class_dist, val_edema_class_dist = val_loader.dataset.class_dist()
    val_retino_class_dist = torch.tensor(val_retino_class_dist, dtype=torch.float32, device=device)    
    val_edema_class_dist = torch.tensor(val_edema_class_dist, dtype=torch.float32, device=device)
    
    
    # Define Loss Functions
    focal_loss_fn = { 
                     'train': loss.FocalRetinalDiseaseLoss(gamma=args.gamma, alpha_retino=1 - train_retino_class_dist, alpha_edema=train_edema_class_dist[0].item()),
                      'test': loss.FocalRetinalDiseaseLoss(gamma=args.gamma, alpha_retino=1 - test_retino_class_dist, alpha_edema=test_edema_class_dist[0].item()),
                      'val': loss.FocalRetinalDiseaseLoss(gamma=args.gamma, alpha_retino=1 - val_retino_class_dist, alpha_edema=val_edema_class_dist[0].item()),
                    }
    
    
    # normal_loss_fn = loss.NormalRetinalDiseaseLoss(alpha_retino=1 - train_retino_class_dist)
    normal_loss_fn = loss.NormalRetinalDiseaseLoss()
    
    
    # Define Model
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device=device)
    
    
    # Define Optimizer
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=args.learning_rate, 
                                 weight_decay=args.weight_decay, 
                                 betas=(args.momentum, args.momentum ** 2),
                                 )


    # Load Checkpoint
    start_epoch = 0
    if args.starting_point is not None:
        start_epoch = load_model(args.starting_point, model, optimizer)
    
    
    # Load statistics
    stats = load_stats(os.path.join(args.output, 'statistics.json'))
    
    
    # Train Model
    print('Training Model...')
    for e in range(start_epoch, args.epochs):
        
        train_loss = train(model, train_loader, optimizer, focal_loss_fn["train"], args.step_inv, args.batch_print_inv, device)
        
        val_loss_fns = [normal_loss_fn, focal_loss_fn['val']]
        val_result = test(model, val_loader, val_loss_fns, device)
        
        print(f"Epoch {e+1}\n-------------------------------")
        print_test_result(val_result=val_result)
        
        stats.append([e, {'train_loss': train_loss, 'val_result': val_result}])
        
        if (e + 1) % args.save_inv == 0:
            save_checkpoint(os.path.join(args.output, f'checkpoint_{str(e + 1)}.pth'), model, optimizer, e, train_loss, val_result.loss[1])
    
    
    # Save the model
    print("Saving Model...")
    torch.save(model.state_dict(), os.path.join(args.output, 'final_model.pth'))
    
    
    # Save statistics
    with open(os.path.join(args.output, 'statistics.json'), 'w') as f:
        json.dump(stats, f, cls=TestResulutJsonEncoder, indent=4)
    
    
    test_loss_fns = [normal_loss_fn, focal_loss_fn['test']]
    test_result = test(model, test_loader, test_loss_fns, device)
    
    print("Final Test Result:")
    val_result = print_test_result(val_result=val_result)