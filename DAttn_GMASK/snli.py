from collections import Counter, defaultdict
from copy import deepcopy
from tqdm import tqdm
from typing import Dict, List, Set
import json
import torch

from text import TextField
from data_utils import split_data, text_to_sentences

import jsonlines
from collections import defaultdict
import random
from abc import abstractmethod


class Dataset:
    def __init__(self,
                 ids: Set[str],
                 id_to_document: Dict[str, List[torch.LongTensor]],
                 id_mapping: Dict[str, Dict[str, Set[str]]],
                 negative_ids: Set[id] = None,
                 label_map: Dict[str, int] = None,
                 evidence: Dict[str,list] = None,
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
        # print(f'Vocabulary size = {len(self.text_field.vocabulary):,}')
        print()


class SNLIDataLoader(DataLoader):
    def __init__(self, args):
        """Loads the pubmed dataset."""

        # Determine word to index mapping
        self.small_data = args.small_data

        # Load data
        train_label, train_evidences = self.load_label(
            args.data_path, "train", args.small_data
        )
        dev_label, dev_evidences = self.load_label(
            args.data_path, "val", args.small_data
        )
        test_label, test_evidences = self.load_label(
            args.data_path, "test", args.small_data
        )

        train_id_mapping = {
            k + "_premise": k + "_hypothesis" for k in train_label.keys()
        }
        dev_id_mapping = {k + "_premise": k + "_hypothesis" for k in dev_label.keys()}
        test_id_mapping = {k + "_premise": k + "_hypothesis" for k in test_label.keys()}
        train_ids = set(train_label.keys())
        dev_ids = set(dev_label.keys())
        test_ids = set(test_label.keys())

        if self.small_data:
            allids = (
                list(train_id_mapping.keys())
                + list(train_id_mapping.values())
                + list(dev_id_mapping.keys())
                + list(dev_id_mapping.values())
                + list(test_id_mapping.keys())
                + list(test_id_mapping.values())
            )
            id_to_text = self.load_text(args.data_path)
            id_to_text = {k: v for k, v in id_to_text.items() if k in allids}
        else:
            id_to_text = self.load_text(args.data_path)

        texts = list(id_to_text.values())
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
        )
        self._dev = Dataset(
            ids=dev_ids,
            id_to_document=id_to_doctoken,
            id_mapping=dev_id_mapping,
            label_map=dev_label,
            evidence=dev_evidences,
        )
        self._test = Dataset(
            ids=test_ids,
            id_to_document=id_to_doctoken,
            id_mapping=test_id_mapping,
            label_map=test_label,
            evidence=test_evidences,
        )

        # self.print_stats()

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
            idx = line["annotation_id"]
            labels[idx] = label_toi[label]
            evidences[label + "_hypothesis"] = []
            evidences[label + "_premise"] = []
            for evi in line["evidences"][0]:
                evidences[evi["docid"]].append((evi["start_token"], evi["end_token"]))
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
