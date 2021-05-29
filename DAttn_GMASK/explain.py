import argparse
import torch
from torch import nn
from torch.optim import Adam
import os
import numpy as np
import random
from load_data import LoadData
from deatten_model import DeAttenModel
from group_mask import interpret
from model_utils import NoamLR
import matplotlib.pyplot as plt
import seaborn as sns


os.environ["CUDA_VISIBLE_DEVICES"]="1"
parser = argparse.ArgumentParser(description='Decomposable Attention Model')
parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--clip', type=float, default=0.1, help='gradient clipping')
parser.add_argument('--epochs', type=int, default=8, help='number of epochs for training')
parser.add_argument('--batch-size', type=int, default=1, help='batch size for training')
parser.add_argument('--dropout', type=float, default=0.2, help='the probability for dropout')
parser.add_argument('--small_data', type=bool, default=False, help='load small data for debugging')
parser.add_argument('--data_path', type=str, default="./esnli/docs.jsonl", help='data path')
parser.add_argument('--embedding_path', type=str, default="./fastText/cc.en.300.bin", help='embedding path')
parser.add_argument('--hidden_size', type=int, default=300, help='Hidden size of the network')
parser.add_argument('--mask_hidden_dim', type=int, default=300, help='Hidden size of the Mask layer')
parser.add_argument('--ffn_hidden_size', type=int, default=300, help='hidden size in decomposable attention model')
parser.add_argument('--embed-dim', type=int, default=300, help='number of embedding dimension')
parser.add_argument("--max_sent_len", type=int, dest="max_sent_len", default=150, help='max sentence length: 250 for padding, 2525 for non-padding')
parser.add_argument('--save', type=str, default='deatten.pt', help='path to save the final model')
parser.add_argument('--gpu', default=0, type=int, help='0:gpu, -1:cpu')
parser.add_argument('--gpu_id', default='0', type=str, help='gpu id')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
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


def test_explain(model, flavor):
    model.eval()
    fileobject = open('interpretation_test.txt', 'w')
    fileobject1 = open('sort_index_test.txt', 'w')

    sampler = dev_sampler if flavor == "eval" else test_sampler
    wordvocab = list(sampler.text_field.vocabulary.keys())
    acc, size, count = 0, 0, 0
    for batch in sampler():
        random_seed()
        count += 1
        print(count)

        fileobject.write(str(count))
        fileobject.write('\n')
        fileobject1.write(str(count))
        fileobject1.write('\n')
        input_words = []
        for btxt in batch[0].data[0]:
            if (wordvocab[btxt] != '<pad>' and wordvocab[btxt] != '<unk>'):
                input_words.append(wordvocab[btxt])
                fileobject.write(wordvocab[btxt])
                fileobject.write(' ')
                fileobject1.write(wordvocab[btxt])
                fileobject1.write(' ')
        fileobject.write(' **|**  ')
        fileobject1.write(' **|**  ')

        for btxt in batch[0].data[1]:
            if (wordvocab[btxt] != '<pad>' and wordvocab[btxt] != '<unk>'):
                input_words.append(wordvocab[btxt])
                fileobject.write(wordvocab[btxt])
                fileobject.write(' ')
                fileobject1.write(wordvocab[btxt])
                fileobject1.write(' ')
        fileobject.write(' >> ')
        fileobject1.write(' >> ')

        with torch.no_grad():
            targets, pred = model(*batch)
            targets = [t for target in targets for t in target["targets"]]
            targets = torch.stack(targets)

            _, pred = pred.max(dim=1)         

        fileobject.write(str(targets.cpu().numpy()[0]))
        fileobject1.write(str(targets.cpu().numpy()[0]))
        fileobject.write(' >> ')
        fileobject1.write(' >> ')

        fileobject.write(str(pred.cpu().numpy()[0]))
        fileobject.write(' >> ')
        fileobject1.write(str(pred.cpu().numpy()[0]))
        fileobject1.write(' >> ')

        # explain model prediction
        args.input_words = input_words
        output, z_prob, t_prob, x_ids = interpret(args, model, batch, pred)
        acc += (output == pred).sum().float()
        size += len(pred)
        sent1_len = (batch[0][0] != 0).nonzero()[-1, 0] + 1
        for i, v in enumerate(x_ids):
            if v >= sent1_len:
                x_ids[i] += batch[0][0].shape[0] - sent1_len

        batch_text = torch.cat((batch[0].data[0], batch[0].data[1]), dim=0)
        for idx in x_ids:
            fileobject.write(wordvocab[batch_text[int(idx)]])
            fileobject.write(' ')
            fileobject1.write(str(int(idx)))
            fileobject1.write(' ')
        fileobject.write(' >> ')
        fileobject1.write(' >> ')
        fileobject.write(str(output.cpu().numpy()[0]))
        fileobject.write('\n')
        fileobject1.write(str(output.cpu().numpy()[0]))
        fileobject1.write('\n')

    acc /= size

    return acc


if __name__ == "__main__":
    if args.gpu > -1:
        args.device = "cuda"
    else:
        args.device = "cpu"

    # load data
    text_field, train_sampler, dev_sampler, test_sampler = LoadData(args)   

    random_seed()

    # test
    if args.gpu > -1:
        with open(args.save, 'rb') as f:
            model = torch.load(f)
        model.to(torch.device(args.device))
    else:
        with open(args.save, 'rb') as f:
            model = torch.load(f, map_location='cpu')
    acc = test_explain(model, 'test')
    print('\nacc {:.6f}'.format(acc))
