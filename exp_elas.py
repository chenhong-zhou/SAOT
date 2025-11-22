import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import *
from utils.testloss import TestLoss
from model_dict import get_model
from utils.normalizer import UnitTransformer

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser('Training SAOT')

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--model', type=str, default='SAOT_1D')
parser.add_argument('--n-hidden', type=int, default=64, help='hidden dim')
parser.add_argument('--n-layers', type=int, default=3, help='layers')
parser.add_argument('--n-heads', type=int, default=4)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument("--gpu", type=str, default='1', help="GPU index to use")
parser.add_argument('--max_grad_norm', type=float, default=None)
parser.add_argument('--downsample', type=int, default=5)
parser.add_argument('--mlp_ratio', type=float, default=1.0) 
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--ntrain', type=int, default=1000)
parser.add_argument('--unified_pos', type=int, default=0)
parser.add_argument('--ref', type=int, default=8)
parser.add_argument('--slice_num', type=int, default=32)
parser.add_argument('--eval', type=int, default=0)
parser.add_argument('--save_name', type=str, default='elas_SAOT')
parser.add_argument('--data_path', type=str, default='./geo-FNO')

args = parser.parse_args()
eval = args.eval
save_name = args.save_name

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params


def main():
    ntrain = args.ntrain
    ntest = 200
    PATH_Sigma = args.data_path + '/elasticity/Meshes/Random_UnitCell_sigma_10.npy'
    PATH_XY = args.data_path + '/elasticity/Meshes/Random_UnitCell_XY_10.npy'

    input_s = np.load(PATH_Sigma)  
    input_s = torch.tensor(input_s, dtype=torch.float).permute(1, 0) 
    input_xy = np.load(PATH_XY)    
    input_xy = torch.tensor(input_xy, dtype=torch.float).permute(2, 0, 1) 

    train_s = input_s[:ntrain]
    test_s = input_s[-ntest:]
    train_xy = input_xy[:ntrain]
    test_xy = input_xy[-ntest:]
    print(input_s.shape, input_xy.shape) 
    print(train_xy.shape, test_xy.shape, train_s.shape, test_s.shape) 
    y_normalizer = UnitTransformer(train_s)

    train_s = y_normalizer.encode(train_s)
    y_normalizer.cuda()

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_xy, train_xy, train_s),
                                               batch_size=args.batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_xy, test_xy, test_s),
                                              batch_size=args.batch_size,
                                              shuffle=False)

    print("Dataloading is over.")

    model = get_model(args).Model(space_dim=2,
                                  n_layers=args.n_layers,
                                  n_hidden=args.n_hidden,
                                  dropout=args.dropout,
                                  n_head=args.n_heads,
                                  Time_Input=False,
                                  mlp_ratio=args.mlp_ratio,
                                  fun_dim=0,
                                  out_dim=1,
                                  slice_num=args.slice_num,
                                  ref=args.ref,
                                  unified_pos=args.unified_pos,
                                  is_filter=False).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(args)
    print(model)
    count_parameters(model)

    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,  T_max=args.epochs) 

    myloss = TestLoss(size_average=False)

    if eval:
        model.load_state_dict(torch.load("./checkpoints/" + save_name + ".pt"))
        model.eval()
        if not os.path.exists('./results/' + save_name + '/'):
            os.makedirs('./results/' + save_name + '/')
        rel_err = 0.0

        with torch.no_grad():
            for pos, fx, y in test_loader:

                x, fx, y = pos.cuda(), fx.cuda(), y.cuda()
                out = model(x, None).squeeze(-1)
                out = y_normalizer.decode(out)
                tl = myloss(out, y).item()
                rel_err += tl
               
        rel_err /= ntest
        print("rel_err : {}".format(rel_err)) 
    else:
        for ep in range(args.epochs):

            model.train()
            train_loss = 0

            for pos, fx, y in train_loader:

                x, fx, y = pos.cuda(), fx.cuda(), y.cuda()  
                optimizer.zero_grad()
                out = model(x, None).squeeze(-1)
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)
                loss = myloss(out, y)
                loss.backward()

                if args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                train_loss += loss.item()
            scheduler.step()

            train_loss = train_loss / ntrain
            print("Epoch {} Train loss : {:.5f}".format(ep, train_loss))
            
            model.eval()
            rel_err = 0.0
            with torch.no_grad():
                for pos, fx, y in test_loader:
                    x, fx, y = pos.cuda(), fx.cuda(), y.cuda()
                    out = model(x, None).squeeze(-1)
                    out = y_normalizer.decode(out)
                    tl = myloss(out, y).item()
                    rel_err += tl

            rel_err /= ntest
            print("rel_err : {}".format(rel_err))


        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        print('save model')
        torch.save(model.state_dict(), os.path.join('./checkpoints', save_name + '.pt'))
    
    print('count_parameters(model): ', count_parameters(model))
    print('filename: ', os.path.basename(__file__))

if __name__ == "__main__":
    main()
