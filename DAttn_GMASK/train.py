import argparse
import torch
from torch import nn
from torch.optim import Adam
import os
import numpy as np
import random
from load_data import LoadData
from deatten_model import DeAttenModel
from model_utils import NoamLR
import matplotlib.pyplot as plt


os.environ["CUDA_VISIBLE_DEVICES"]="1"
parser = argparse.ArgumentParser(description='Decomposable Attention Model')
parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--clip', type=float, default=0.1, help='gradient clipping')
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs for training')
parser.add_argument('--batch-size', type=int, default=400, help='batch size for training')
parser.add_argument('--dropout', type=float, default=0.2, help='the probability for dropout')
parser.add_argument('--small_data', type=bool, default=False, help='load small data for debugging')
parser.add_argument('--data_path', type=str, default="./esnli/docs.jsonl", help='data path')
parser.add_argument('--embedding_path', type=str, default="/localtmp/hc9mx/fastText/cc.en.300.bin", help='embedding path')
parser.add_argument('--hidden_size', type=int, default=300, help='Hidden size of the network')
parser.add_argument('--ffn_hidden_size', type=int, default=300, help='hidden size in decomposable attention model')
parser.add_argument('--embed-dim', type=int, default=300, help='number of embedding dimension')
parser.add_argument("--max_sent_len", type=int, dest="max_sent_len", default=150, help='max sentence length: 250 for padding, 2525 for non-padding')
parser.add_argument('--save', type=str, default='deatten.pt', help='path to save the final model')
parser.add_argument('--gpu', default=0, type=int, help='0:gpu, -1:cpu')
parser.add_argument('--gpu_id', default='0', type=str, help='gpu id')
parser.add_argument('--seed', type=int, default=712, help='random seed')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)


def random_seed():
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return


def evaluate_model(model, flavor):
    model.eval()
    with torch.no_grad():
        sampler = dev_sampler if flavor == "eval" else test_sampler
        acc, loss, size, count = 0, 0, 0, 0
        for batch in sampler():
            count += 1

            targets, pred = model(*batch)
            targets = [t for target in targets for t in target["targets"]]
            targets = torch.stack(targets)

            batch_loss= loss_fn(pred, targets)
            loss += batch_loss.item()

            _, pred = pred.max(dim=1)
            acc += (pred == targets).sum().float()
            size += len(pred)

        acc /= size
        loss /= count

    return acc, loss


def train_model(model):

    best_dev_acc = None
    train_accs = []
    dev_accs = []
    eps = []

    for epoch in range(1, args.epochs+1):
        eps.append(epoch)
        print("\n## The {} Epoch, All {} Epochs ! ##".format(epoch, args.epochs))
        model.train()
        optimizer.zero_grad()

        trn_size, trn_corrects, trn_loss, count = 0, 0, 0, 0

        with torch.enable_grad():
            for i, batch in enumerate(train_sampler()):
                count += 1

                targets, pred = model(*batch)
                targets = [t for target in targets for t in target["targets"]]
                targets = torch.stack(targets)

                batch_loss= loss_fn(pred, targets)
                trn_loss += batch_loss.item()

                # Optimize
                batch_loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()  # Update learning rate schedule
                optimizer.zero_grad()

                _, pred = pred.max(dim=1)
                trn_corrects += (pred == targets).sum().float()
                trn_size += len(pred)

        # evaluation
        dev_acc, dev_loss = evaluate_model(model, 'eval')
        if not best_dev_acc or dev_acc > best_dev_acc:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_dev_acc = dev_acc
        else:
            scheduler.step()

        train_acc = trn_corrects / trn_size
        train_loss = trn_loss / count
        train_accs.append(train_acc.cpu().numpy())
        dev_accs.append(dev_acc.cpu().numpy())

        print('local_epoch {} | train_loss {:.6f} | train_acc {:.6f} | dev_loss {:.6f} | dev_acc {:.6f} | best_dev_acc {:.6f}'.format( \
                epoch, train_loss, train_acc, dev_loss, dev_acc, best_dev_acc))

    plt.figure()
    plt.plot(eps, train_accs, label='training_acc')
    plt.plot(eps, dev_accs, label='dev_acc')
    plt.xlabel("epochs")
    plt.ylabel("acc")
    plt.title("Training and validation acc")
    plt.legend()
    # plt.show()
    plt.savefig("train_dev.png")
    
    # test
    del model
    with open(args.save, 'rb') as f:
        model = torch.load(f)
    model.to(torch.device(args.device))
    test_acc, test_loss = evaluate_model(model, 'test')
    print('\nfinal_test_acc {:.6f}'.format(test_acc))


if __name__ == "__main__":
    if args.gpu > -1:
        args.device = "cuda"
    else:
        args.device = "cpu"

    # load data
    text_field, train_sampler, dev_sampler, test_sampler = LoadData(args)   

    model = DeAttenModel(args, text_field)   
    model.to(torch.device(args.device))

    # metric
    loss_fn = nn.CrossEntropyLoss()

    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = NoamLR(optimizer, warmup_steps=100, model_size=model.output_size, last_epoch=-1)

    random_seed()
    # train_model(model)

    with open(args.save, 'rb') as f:
        model = torch.load(f)
    model.to(torch.device(args.device))
    test_acc, test_loss = evaluate_model(model, 'test')
    print('\nfinal_test_acc {:.6f}'.format(test_acc))
