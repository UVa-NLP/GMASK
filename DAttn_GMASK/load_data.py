from snli import SNLIDataLoader
from snli_sampler import SNLISampler as Sampler
import dill

def LoadData(args):
    data_loader = SNLIDataLoader(args)
    train_sampler = Sampler(
        data=data_loader.train,
        text_field=data_loader.text_field,
        batch_size=args.batch_size,
        shuffle=True,
        num_positives=None,
        num_negatives=None,
        resample_negatives=True,
        device=args.device,
    )

    dev_sampler = Sampler(
        data=data_loader.dev,
        text_field=data_loader.text_field,
        batch_size=args.batch_size,
        num_positives=None,
        num_negatives=None,
        device=args.device,
    )

    test_sampler = Sampler(
        data=data_loader.test,
        text_field=data_loader.text_field,
        batch_size=args.batch_size,
        num_positives=None,
        num_negatives=None,
        device=args.device,
    )

    return data_loader.text_field, train_sampler, dev_sampler, test_sampler