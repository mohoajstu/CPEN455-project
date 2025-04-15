import time
import os
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import wandb
from utils import *
from model import conditional_pixelcnn, random_classifier
from dataset import *
from tqdm import tqdm
from pprint import pprint
import argparse
from pytorch_fid.fid_score import calculate_fid_given_paths

def train_or_test(model, data_loader, optimizer, loss_op, device, args, epoch, mode='training'):
    if mode == 'training':
        model.train()
    else:
        model.eval()

    deno = args.batch_size * np.prod(args.obs) * np.log(2.)
    loss_tracker = mean_tracker()

    for batch_idx, item in enumerate(tqdm(data_loader)):
        model_input, cat_name = item 
        model_input = model_input.to(device)

        label_id = []
        for cname in cat_name:
            if cname in my_bidict:
                label_id.append(my_bidict[cname])
            else:
                label_id.append(0) 
        label_id = torch.LongTensor(label_id).to(device)

        model_output = model(model_input, label=label_id, sample=False)
        loss = loss_op(model_input, model_output)

        loss_tracker.update(loss.item() / deno)

        if mode == 'training':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if args.en_wandb:
        wandb.log({mode + "-Average-BPD": loss_tracker.get_mean()})
        wandb.log({mode + "-epoch": epoch})

def get_label(model, model_input, device):
    model.eval()
    B = model_input.size(0)
    all_ll = torch.zeros(B, args.num_classes, device=device)

    with torch.no_grad():
        for c in range(args.num_classes):
            label_vec = torch.full((B,), c, dtype=torch.long, device=device)
            out = model(model_input, label=label_vec, sample=False)
            for i in range(B):
                loss_val = discretized_mix_logistic_loss(
                    model_input[i:i+1], out[i:i+1])
                all_ll[i, c] = -loss_val

    predicted_label = torch.argmax(all_ll, dim=1)
    return predicted_label

def evaluate_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    for batch_idx, item in enumerate(data_loader):
        model_input, cat_name = item
        model_input = model_input.to(device)
        label_id = [my_bidict.get(name, 0) for name in cat_name]
        label_id = torch.tensor(label_id).to(device)
        pred = get_label(model, model_input, device)
        correct += (pred == label_id).sum().item()
        total += model_input.size(0)
    return correct / total if total > 0 else 0.0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-w','--en_wandb', type=bool, default=False,
                        help='Enable wandb logging')
    parser.add_argument('-t','--tag', type=str, default='default',
                        help='Tag for this run')

    parser.add_argument('-c','--sampling_interval', type=int, default=5,
                        help='sampling interval')
    parser.add_argument('-i','--data_dir', type=str,
                        default='data', help='Location for the dataset')
    parser.add_argument('-o','--save_dir', type=str, default='models',
                        help='Location for parameter checkpoints and samples')
    parser.add_argument('-sd','--sample_dir', type=str, default='samples',
                        help='Location for saving samples')
    parser.add_argument('-d','--dataset', type=str,
                        default='cpen455', help='cifar|mnist|cpen455')
    parser.add_argument('-st','--save_interval', type=int, default=10,
                        help='Epoch interval to checkpoint')
    parser.add_argument('-r','--load_params', type=str, default=None,
                        help='Restore training from checkpoint?')
    parser.add_argument('--obs', type=tuple, default=(3,32,32),
                        help='Observation shape')

    parser.add_argument('-q','--nr_resnet', type=int, default=1,
                        help='Number of residual blocks per stage')
    parser.add_argument('-n','--nr_filters', type=int, default=40,
                        help='Number of filters (width) in model')
    parser.add_argument('-m','--nr_logistic_mix', type=int, default=5,
                        help='Number of logistic components in the mixture')
    parser.add_argument('-l','--lr', type=float, default=0.0002,
                        help='Base learning rate')
    parser.add_argument('-e','--lr_decay', type=float, default=0.999995,
                        help='Learning rate decay per step')
    parser.add_argument('-b','--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('-sb','--sample_batch_size', type=int, default=32,
                        help='Batch size during sampling')
    parser.add_argument('-x','--max_epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('-s','--seed', type=int, default=1,
                        help='Random seed')

    parser.add_argument('--num_classes', type=int, default=4,
                        help='Number of classes for cpen455 dataset')

    args = parser.parse_args()
    pprint(args.__dict__)

    check_dir_and_create(args.save_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model_name = f'pcnn_{args.dataset}_'
    model_path = args.save_dir + '/'
    if args.load_params is not None:
        model_name += 'load_model'
        model_path += model_name + '/'
    else:
        model_name += 'from_scratch'
        model_path += model_name + '/'

    job_name = f"PCNN_Training_dataset:{args.dataset}_{args.tag}"

    if args.en_wandb:
        wandb.init(project="CPEN455HW", name=job_name)
        wandb.config.current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        wandb.config.update(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kwargs = {'num_workers':0, 'pin_memory':True, 'drop_last':True}

    if "cpen455" in args.dataset:
        ds_transforms = transforms.Compose([transforms.Resize((32,32)), rescaling])
        train_loader = torch.utils.data.DataLoader(
            CPEN455Dataset(root_dir=args.data_dir, mode='train', transform=ds_transforms),
            batch_size=args.batch_size, shuffle=True, **kwargs
        )
        test_loader  = torch.utils.data.DataLoader(
            CPEN455Dataset(root_dir=args.data_dir, mode='test', transform=ds_transforms),
            batch_size=args.batch_size, shuffle=True, **kwargs
        )
        val_loader  = torch.utils.data.DataLoader(
            CPEN455Dataset(root_dir=args.data_dir, mode='validation', transform=ds_transforms),
            batch_size=args.batch_size, shuffle=True, **kwargs
        )
    else:
        raise ValueError("This script is set up primarily for the cpen455 dataset.")

    args.obs = (3,32,32)

    loss_op   = lambda real, fake: discretized_mix_logistic_loss(real, fake)
    sample_op = lambda x: sample_from_discretized_mix_logistic(x, args.nr_logistic_mix)

    model = conditional_pixelcnn(
        nr_resnet=args.nr_resnet,
        nr_filters=args.nr_filters,
        nr_logistic_mix=args.nr_logistic_mix,
        input_channels=3,
        nr_classes=args.num_classes,   
        emb_dim=args.nr_filters       
    ).to(device)

    if args.load_params:
        model.load_state_dict(torch.load(args.load_params))
        print("Model parameters loaded from", args.load_params)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)

    for epoch in range(args.max_epochs):
      print(f"\nEpoch {epoch+1}/{args.max_epochs}")
      train_or_test(model, train_loader, optimizer, loss_op, device, args, epoch, mode='training')
      scheduler.step()
      
      acc_test = evaluate_accuracy(model, test_loader, device)
      acc_val = evaluate_accuracy(model, val_loader, device)
      print(f"🧪 Accuracy (Test): {acc_test:.4f} | Accuracy (Val): {acc_val:.4f}")
      if args.en_wandb:
          wandb.log({
              "test_accuracy": acc_test,
              "val_accuracy": acc_val,
              "epoch": epoch+1
          })

      if epoch % args.sampling_interval == 0:
          print('......sampling......')
          model.eval()
          with torch.no_grad():
              sample_t = sample(model, args.sample_batch_size, args.obs, sample_op)
          sample_t = rescaling_inv(sample_t)
          save_images(sample_t, args.sample_dir, label=f"epoch{epoch+1}")
          
          gen_path = args.sample_dir
          ref_path = os.path.join(args.data_dir, 'test')
          paths = [gen_path, ref_path]
          
          try:
              fid_score = calculate_fid_given_paths(paths, batch_size=32, device=device, dims=192)
              print("FID score:", fid_score)
          except Exception as e:
              fid_score = 999999 
              print("FID calculation failed with error:", e)

          if args.en_wandb:
              wandb.log({
                  "sample_images": wandb.Image(sample_t, caption=f"Epoch {epoch+1}"),
                  "FID": fid_score,
                  "epoch": epoch+1
              })
              
      if (epoch+1) % args.save_interval == 0:
          if not os.path.exists("models"):
              os.makedirs("models")
          ckpt_path = f'models/{model_name}_{epoch}.pth'
          torch.save(model.state_dict(), ckpt_path)
          print("Saved model checkpoint:", ckpt_path)
