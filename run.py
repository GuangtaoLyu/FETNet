import argparse
import os
from model import FETNetModel
from dataset import Dataset
from torch.utils.data import DataLoader
import torch

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_root', type=str,default='')
    parser.add_argument('--mask_root', type=str,default='')
    parser.add_argument('--text_root', type=str,default='')
    parser.add_argument('--model_save_path', type=str, default='checkpoint')
    parser.add_argument('--result_save_path', type=str, default='results')
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--model_path_g', type=str, default="checkpoint/g_enstext.pth")
    parser.add_argument('--model_path_d', type=str, default="checkpoint/xxx.pth")
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--n_threads', type=int, default=3)
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--gpu_id', type=str, default="0")
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    model = FETNetModel()
    if args.test:
        model.initialize_model(args.model_path_g, args.model_path_d,False)
        total = sum([param.nelement() for param in model.G.parameters()])
        print("Number of G'parameter: %.2fM" % (total / 1e6))
        total2 = sum([param.nelement() for param in model.D.parameters()])
        print("Number of D'parameter: %.2fM" % (total2 / 1e6))
        print("Number of parameter: %.2fM" % ((total+total2) / 1e6))
        model.cuda()
        dataloader = DataLoader(Dataset(args.text_root, args.mask_root, args.gt_root,mask_reverse = True, training=False))
        model.test(dataloader, args.result_save_path)
    else:
        model.initialize_model(args.model_path_g,args.model_path_d, True)
        model.cuda()
        dataloader = DataLoader(Dataset(args.text_root, args.mask_root, args.gt_root,mask_reverse = True), batch_size = args.batch_size, shuffle = True, num_workers = args.n_threads,drop_last=True,pin_memory=True)
        model.train(dataloader, args.model_save_path, args.finetune, args.num_epochs)

if __name__ == '__main__':
    run()
