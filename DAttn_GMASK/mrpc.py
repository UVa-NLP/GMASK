from collections import Counter, defaultdict
from copy import deepcopy
from tqdm import tqdm
from typing import Dict, List, Set
import json
import torch
import csv
import sys

from text import TextField
from data_utils import split_data, text_to_sentences

import jsonlines
from collections import defaultdict
import random
from abc import abstractmethod

def read_tsv(data_path):
    lines = []
    with open(data_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            a = line.split('\t')
            lines.append(a)
    return lines


class Dataset:
    def __init__(self,
                 ids: Set[str],
                 id_to_document: Dict[str, List[torch.LongTensor]],
                 id_mapping: Dict[str, Dict[str, Set[str]]],
                 negative_ids: Set[id] = None,
                 label_map: Dict[str, int] = None,
                 evidence: Dict[str,list] = None,
                 idnum: Dict[str,list] = None,
                 id_to_sentlength: Dict[str,list] = None):
        """
        :param ids: A set of ids from which to sample during training.
        Note: May not contain all ids since some ids should not be sampled.
        :param id_to_document: A dictionary mapping ids to a dictionary
        which maps "sentences" to the sentences in the document.
        :param id_mapping: A dictionary mapping ids to a dictionary which maps
        "similar" to similar ids and "dissimilar" to dissimilar ids.
        :param negative_ids: The set of ids which can be sampled as negatives.
        If None, any id can be sampled as a negative.
        :param id_to_sentlength: save the length of sentences in document. Only used in multiRC with bert model.
        """
        self.id_set = ids
        self.id_list = sorted(self.id_set)
        self.id_to_document = id_to_document
        self.id_mapping = id_mapping
        self.negative_ids = negative_ids or self.id_set
        self.label_map = label_map
        self.evidence = evidence
        self.idnum = idnum
        self.id_to_sentlength = id_to_sentlength

    def __len__(self) -> int:
        return len(self.id_set)


class DataLoader:
    @property
    @abstractmethod
    def train(self) -> Dataset:
        """Returns the training data."""
        pass

    @property
    @abstractmethod
    def dev(self) -> Dataset:
        """Returns the validation data."""
        pass

    @property
    @abstractmethod
    def test(self) -> Dataset:
        """Returns the test data."""
        pass

    @property
    @abstractmethod
    def text_field(self) -> TextField:
        """Returns the text field."""
        pass

    def print_stats(self) -> None:
        """Prints statistics about the data."""
        print()
        print(f'Total size = {len(self.train) + len(self.dev) + len(self.test):,}')
        print()
        print(f'Train size = {len(self.train):,}')
        print(f'Dev size = {len(self.dev):,}')
        print(f'Test size = {len(self.test):,}')
        print()
        print()


class MRPCDataLoader(DataLoader):
    def __init__(self, args):
        """Loads the pubmed dataset."""

        # Determine word to index mapping
        self.small_data = args.small_data

        texts = []
        id_to_text = {}
        id_to_text_train, train_id_mapping, train_label, train_ids, train_idnum = self.load_dt(args.data_path, "train")
        id_to_text_dev, dev_id_mapping, dev_label, dev_ids, dev_idnum = self.load_dt(args.data_path, "dev", len(train_ids))
        id_to_text_test, test_id_mapping, test_label, test_ids, test_idnum = self.load_dt(args.data_path, "test", len(train_ids) + len(dev_ids))
        
        texts.extend(list(id_to_text_train.values()))
        texts.extend(list(id_to_text_dev.values()))
        texts.extend(list(id_to_text_test.values()))
        id_to_text.update(id_to_text_train)
        id_to_text.update(id_to_text_dev)
        id_to_text.update(id_to_text_test)
        
        train_evidences, dev_evidences, test_evidences = {}, {}, {}
        self._text_field = TextField()
        self._text_field.build_vocab(texts)

        # Convert sentences to indices
        id_to_doctoken: Dict[str, List[torch.LongTensor]] = {
            idx: self._text_field.process(text)
            # for id, sentence in tqdm(id_to_doctoken.items())
            for idx, text in tqdm(id_to_text.items())
        }
        

        # Define train, dev, test datasets
        self._train = Dataset(
            ids=train_ids,
            id_to_document=id_to_doctoken,
            id_mapping=train_id_mapping,
            label_map=train_label,
            evidence=train_evidences,
            idnum = train_idnum,
        )
        self._dev = Dataset(
            ids=dev_ids,
            id_to_document=id_to_doctoken,
            id_mapping=dev_id_mapping,
            label_map=dev_label,
            evidence=dev_evidences,
            idnum = dev_idnum,
        )
        self._test = Dataset(
            ids=test_ids,
            id_to_document=id_to_doctoken,
            id_mapping=test_id_mapping,
            label_map=test_label,
            evidence=test_evidences,
            idnum = test_idnum,
        )

        # self.print_stats()

    @staticmethod
    def load_dt(path, flavor, offset=0):
        if flavor == 'test':
            label_path = path.replace("train.tsv", flavor+'.tsv')
            lines = read_tsv(label_path)
            id_to_text = defaultdict(dict)
            id_mapping = {}
            labels = defaultdict(dict)
            ids = []
            idnum = defaultdict(dict)
            for idx, line in enumerate(lines):
                if idx == 0:
                    continue
                k1 = '{}_premise'.format(idx + offset)
                k2 = '{}_hypothesis'.format(idx + offset)
                id_to_text[k1] = line[3]
                id_to_text[k2] = line[4]
                id_mapping[k1] = k2 
                ids.append(str(idx + offset))
                labels[str(idx + offset)] = 0
                idnum[str(idx + offset)] = line[0]
            
        else:        
            label_path = path.replace("train.tsv", flavor+'.tsv')
            lines = read_tsv(label_path)
            id_to_text = defaultdict(dict)
            id_mapping = {}
            labels = defaultdict(dict)
            ids = []
            idnum = defaultdict(dict)
            for idx, line in enumerate(lines):
                if idx == 0:
                    continue
                # if len(line) != 5:
                #     continue
                k1 = '{}_premise'.format(idx + offset)
                k2 = '{}_hypothesis'.format(idx + offset)
                id_to_text[k1] = line[3]
                id_to_text[k2] = line[4]
                id_mapping[k1] = k2 
                # labels.append(int(line[0]))
                ids.append(str(idx + offset))
                labels[str(idx + offset)] = int(line[0])
                idnum[str(idx + offset)] = line[0]
        return id_to_text, id_mapping, labels, ids, idnum


    @staticmethod
    def load_text(path: str, small_data: bool = False) -> List[List[List[List[str]]]]:
        data = defaultdict(dict)
        print(f"reading text from {path}")
        reader = jsonlines.Reader(open(path))

        for line in reader:
            data[line["docid"]] = line["document"]

        return data

    @staticmethod
    def load_label(
        path: str, flavor: str, small_data: bool = False
    ) -> List[List[List[List[str]]]]:
        label_path = path.replace("docs", flavor)
        print(f"reading labels from {label_path}")
        labels = defaultdict(dict)
        evidences = defaultdict(list)
        label_toi = {"entailment": 0, "contradiction": 1, "neutral": 2}

        reader = jsonlines.Reader(open(label_path))
        for line in reader:
            label = line["classification"]
            label = int(line[0])
            idx = line["annotation_id"]
            labels[idx] = label_toi[label]
            evidences[label + "_hypothesis"] = []
            evidences[label + "_premise"] = []
            for evi in line["evidences"][0]:
                evidences[evi["docid"]].append((evi["start_tok en"], evi["end_token"]))
            if small_data and len(labels) > 500:
                break
        return labels, evidences

    @property
    def train(self) -> Dataset:
        return self._train

    @property
    def dev(self) -> Dataset:
        return self._dev

    @property
    def test(self) -> Dataset:
        return self._test

    @property
    def text_field(self) -> TextField:
        return self._text_field
