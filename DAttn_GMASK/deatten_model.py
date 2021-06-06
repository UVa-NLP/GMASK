from functools import partial
import math
from typing import Any, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_utils import unpad_tensors, feed_forward
from encoder import Embedder


class DeAttenModel(nn.Module):
    def __init__(
        self,
        args,
        text_field,
    ):
        super(DeAttenModel, self).__init__()

        # Save values
        self.args = args
        self.device = args.device

        self.embedder = Embedder(
            args=args,
            text_field=text_field,
            device=args.device,
        )

        self.attention_forward = feed_forward(
            self.embedder.output_size,
            self.args.ffn_hidden_size,
            self.args.dropout,
            2,
            "relu",
        )

        compare_input = 2 * self.args.ffn_hidden_size

        self.compare_forward = feed_forward(
            compare_input,
            self.args.ffn_hidden_size,
            self.args.dropout,
            2,
        )

        aggregate_input = 2 * self.args.ffn_hidden_size 

        self.cls_forward = feed_forward(
            aggregate_input,
            self.args.ffn_hidden_size,
            self.args.dropout,
            2,
        )

        cls_input = self.args.ffn_hidden_size

        self.out = nn.Linear(cls_input, 3) # set the output dimension as 2 for the quora/qqp/mrpc datasets

        # This is a special model for SNLI, where the logits is (pos_cost, neg_cost, bias)
        self.word_to_word = True
        self.pad_index = text_field.pad_index()
        self.output_size = self.embedder.output_size

    def forward(self, data, scope, targets):
        embeds = self.embedder(data)  # bs(of sent), seq_len, n_hidden
        encoded = self.attention_forward(embeds)
        mask = (data != self.pad_index).float()  # bs, seq_len

        sent1_idx = torch.cat(([s[0] for s in scope]), dim=0)
        sent2_idx = torch.cat(([s[1] for s in scope]), dim=0)

        sent1_len = torch.index_select(mask, 0, sent1_idx).sum(-1).max().long().item()
        sent1_emb = torch.index_select(embeds, 0, sent1_idx)[:, :sent1_len, :]
        sent1_encoded = torch.index_select(encoded, 0, sent1_idx)[:, :sent1_len, :]  # bs x (n/m)x 2*hidden_size
        
        sent2_len = torch.index_select(mask, 0, sent2_idx).sum(-1).max().long().item()
        sent2_emb = torch.index_select(embeds, 0, sent2_idx)[:, :sent2_len, :]
        sent2_encoded = torch.index_select(encoded, 0, sent2_idx)[:, :sent2_len, :]  # bs x (n/m)x 2*hidden_size

        # attend
        score1 = torch.bmm(sent1_encoded, torch.transpose(sent2_encoded, 1, 2))
        # e_{ij} batch_size x len1 x len2
        prob1 = F.softmax(score1.view(-1, sent2_len), dim=1).view(-1, sent1_len, sent2_len)
        # batch_size x len1 x len2

        score2 = torch.transpose(score1.contiguous(), 1, 2)
        score2 = score2.contiguous()
        # e_{ji} batch_size x len2 x len1
        prob2 = F.softmax(score2.view(-1, sent1_len), dim=1).view(-1, sent2_len, sent1_len)
        # batch_size x len2 x len1
        
        sent1_combine = torch.cat((sent1_emb, torch.bmm(prob1, sent2_emb)), 2)
        # batch_size x len1 x (hidden_size x 2)
        sent2_combine = torch.cat((sent2_emb, torch.bmm(prob2, sent1_emb)), 2)
        # batch_size x len2 x (hidden_size x 2)

        # compare
        g1 = self.compare_forward(sent1_combine.view(-1, 2 * self.args.ffn_hidden_size))
        g2 = self.compare_forward(sent2_combine.view(-1, 2 * self.args.ffn_hidden_size))
        g1 = g1.view(-1, sent1_len, self.args.ffn_hidden_size)
        # batch_size x len1 x hidden_size
        g2 = g2.view(-1, sent2_len, self.args.ffn_hidden_size)
        # batch_size x len2 x hidden_size

        # Aggregate
        sent1_output = torch.sum(g1, 1)  # batch_size x 1 x hidden_size
        sent1_output = torch.squeeze(sent1_output, 1)
        sent2_output = torch.sum(g2, 1)  # batch_size x 1 x hidden_size
        sent2_output = torch.squeeze(sent2_output, 1)

        input_combine = torch.cat((sent1_output, sent2_output), 1)
        # batch_size x (2 * hidden_size)

        h = self.cls_forward(input_combine)
        # batch_size * hidden_size

        # final layer
        logits = self.out(h)

        return targets, logits

    def mask_predict(self, local_bsz, data, x_prime):
        data = data.expand(local_bsz, data.shape[0], data.shape[1])
        data = data.reshape(data.shape[0] * data.shape[1], data.shape[2])
        embeds = x_prime.view(local_bsz * 2, -1, self.args.embed_dim)
        encoded = self.attention_forward(embeds)
        mask = (data != self.pad_index).float()  # bs, seq_len
        sent1_idx = torch.cat(([torch.tensor([2*s], dtype=torch.int64) for s in range(local_bsz)]), dim=0).to(self.device)
        sent2_idx = torch.cat(([torch.tensor([2*s+1], dtype=torch.int64) for s in range(local_bsz)]), dim=0).to(self.device)

        sent1_len = torch.index_select(mask, 0, sent1_idx).sum(-1).max().long().item()
        sent1_emb = torch.index_select(embeds, 0, sent1_idx)[:, :sent1_len, :]
        sent1_encoded = torch.index_select(encoded, 0, sent1_idx)[:, :sent1_len, :]  # bs x (n/m)x 2*hidden_size
        
        sent2_len = torch.index_select(mask, 0, sent2_idx).sum(-1).max().long().item()
        sent2_emb = torch.index_select(embeds, 0, sent2_idx)[:, :sent2_len, :]
        sent2_encoded = torch.index_select(encoded, 0, sent2_idx)[:, :sent2_len, :]  # bs x (n/m)x 2*hidden_size

        # attend
        score1 = torch.bmm(sent1_encoded, torch.transpose(sent2_encoded, 1, 2))
        # e_{ij} batch_size x len1 x len2
        prob1 = F.softmax(score1.view(-1, sent2_len), dim=1).view(-1, sent1_len, sent2_len)
        # batch_size x len1 x len2

        score2 = torch.transpose(score1.contiguous(), 1, 2)
        score2 = score2.contiguous()
        # e_{ji} batch_size x len2 x len1
        prob2 = F.softmax(score2.view(-1, sent1_len), dim=1).view(-1, sent2_len, sent1_len)
        # batch_size x len2 x len1
        
        sent1_combine = torch.cat((sent1_emb, torch.bmm(prob1, sent2_emb)), 2)
        # batch_size x len1 x (hidden_size x 2)
        sent2_combine = torch.cat((sent2_emb, torch.bmm(prob2, sent1_emb)), 2)
        # batch_size x len2 x (hidden_size x 2)

        # compare
        g1 = self.compare_forward(sent1_combine.view(-1, 2 * self.args.ffn_hidden_size))
        g2 = self.compare_forward(sent2_combine.view(-1, 2 * self.args.ffn_hidden_size))
        g1 = g1.view(-1, sent1_len, self.args.ffn_hidden_size)
        # batch_size x len1 x hidden_size
        g2 = g2.view(-1, sent2_len, self.args.ffn_hidden_size)
        # batch_size x len2 x hidden_size

        # Aggregate
        sent1_output = torch.sum(g1, 1)  # batch_size x 1 x hidden_size
        sent1_output = torch.squeeze(sent1_output, 1)
        sent2_output = torch.sum(g2, 1)  # batch_size x 1 x hidden_size
        sent2_output = torch.squeeze(sent2_output, 1)

        input_combine = torch.cat((sent1_output, sent2_output), 1)
        # batch_size x (2 * hidden_size)

        h = self.cls_forward(input_combine)
        # batch_size * hidden_size

        # final layer
        logits = self.out(h)

        return logits


    def degra_predict(self, data, scope, feaidx):
        mask = (data != self.pad_index).float()  # bs, seq_len

        if feaidx != []:
            data = data.view(1, -1)
            mask_vec = data.new_ones(data.shape)
            mask_vec[torch.arange(mask_vec.size(0)), torch.tensor(feaidx)] = 0
            data = data * mask_vec
            data = data.view(2, -1)

        embeds = self.embedder(data)  # bs(of sent), seq_len, n_hidden
        encoded = self.attention_forward(embeds)

        sent1_idx = torch.cat(([s[0] for s in scope]), dim=0)
        sent2_idx = torch.cat(([s[1] for s in scope]), dim=0)

        sent1_len = torch.index_select(mask, 0, sent1_idx).sum(-1).max().long().item()
        sent1_emb = torch.index_select(embeds, 0, sent1_idx)[:, :sent1_len, :]
        sent1_encoded = torch.index_select(encoded, 0, sent1_idx)[:, :sent1_len, :]  # bs x (n/m)x 2*hidden_size
        
        sent2_len = torch.index_select(mask, 0, sent2_idx).sum(-1).max().long().item()
        sent2_emb = torch.index_select(embeds, 0, sent2_idx)[:, :sent2_len, :]
        sent2_encoded = torch.index_select(encoded, 0, sent2_idx)[:, :sent2_len, :]  # bs x (n/m)x 2*hidden_size

        # attend
        score1 = torch.bmm(sent1_encoded, torch.transpose(sent2_encoded, 1, 2))
        # e_{ij} batch_size x len1 x len2
        prob1 = F.softmax(score1.view(-1, sent2_len), dim=1).view(-1, sent1_len, sent2_len)
        # batch_size x len1 x len2

        score2 = torch.transpose(score1.contiguous(), 1, 2)
        score2 = score2.contiguous()
        # e_{ji} batch_size x len2 x len1
        prob2 = F.softmax(score2.view(-1, sent1_len), dim=1).view(-1, sent2_len, sent1_len)
        # batch_size x len2 x len1
        
        sent1_combine = torch.cat((sent1_emb, torch.bmm(prob1, sent2_emb)), 2)
        # batch_size x len1 x (hidden_size x 2)
        sent2_combine = torch.cat((sent2_emb, torch.bmm(prob2, sent1_emb)), 2)
        # batch_size x len2 x (hidden_size x 2)

        # compare
        g1 = self.compare_forward(sent1_combine.view(-1, 2 * self.args.ffn_hidden_size))
        g2 = self.compare_forward(sent2_combine.view(-1, 2 * self.args.ffn_hidden_size))
        g1 = g1.view(-1, sent1_len, self.args.ffn_hidden_size)
        # batch_size x len1 x hidden_size
        g2 = g2.view(-1, sent2_len, self.args.ffn_hidden_size)
        # batch_size x len2 x hidden_size

        # Aggregate
        sent1_output = torch.sum(g1, 1)  # batch_size x 1 x hidden_size
        sent1_output = torch.squeeze(sent1_output, 1)
        sent2_output = torch.sum(g2, 1)  # batch_size x 1 x hidden_size
        sent2_output = torch.squeeze(sent2_output, 1)

        input_combine = torch.cat((sent1_output, sent2_output), 1)
        # batch_size x (2 * hidden_size)

        h = self.cls_forward(input_combine)
        # batch_size * hidden_size

        # final layer
        logits = self.out(h)

        return logits
