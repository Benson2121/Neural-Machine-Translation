#! /usr/bin/env python3.7

'''
Authors: Sean Robertson, Jingcheng Niu, Zining Zhu, and Mohamed Abdall
Updated by: Raeid Saqur <raeidsaqur@cs.toronto.edu>

All of the files in this directory and all subdirectories are:
Copyright (c) 2023 University of Toronto
'''

'''All command-line operations in A2'''

import sys
import os
import argparse
import gzip
import random
from typing import List, Sequence, Optional, Union, Any, TextIO
import code
from pathlib import Path

import torch

import a2_dataloader
import a2_encoder_decoder
import a2_training_and_testing

try:
    import wandb
except ImportError:
    pass


def build_vocab(opts: argparse.Namespace):
    word2id = a2_dataloader.build_vocab_from_dir(
        opts.training_dir, opts.lang, opts.max_vocab, opts.min_freq, opts.specials)
    a2_dataloader.write_stoi_to_file(opts.out, word2id)

def build_data_train_dev_split(opts: argparse.Namespace):
    common = a2_dataloader.get_common_prefixes(opts.training_dir)
    random.seed(opts.seed)
    random.shuffle(common)
    if opts.limit:
        common = common[:opts.limit]
    num_train = max(1, int(len(common) * opts.proportion_training))
    train = sorted(common[:num_train])
    dev = sorted(common[num_train:])
    assert not (set(train) & set(dev))
    for prefix_path, prefixes in (
            (opts.train_prefixes, train), (opts.dev_prefixes, dev)):
        with smart_open(prefix_path, 'wt') as open_file:
            open_file.write('\n'.join(prefixes))
            open_file.write('\n')


def init(
        opts: argparse.Namespace,
        dataloader: a2_dataloader.HansardDataLoader
        ) -> a2_encoder_decoder.EncoderDecoder:
    encoder_class = a2_encoder_decoder.Encoder
    if opts.with_attention:
        decoder_class = a2_encoder_decoder.DecoderWithAttention
    elif opts.with_multihead_attention:
        decoder_class = a2_encoder_decoder.DecoderWithMultiHeadAttention
    else:
        decoder_class = a2_encoder_decoder.DecoderWithoutAttention
    return a2_encoder_decoder.EncoderDecoder(
        encoder_class, decoder_class,
        dataloader.dataset.source_vocab_size,
        dataloader.dataset.target_vocab_size,
        dataloader.dataset.source_pad_id,
        dataloader.dataset.target_sos,
        dataloader.dataset.target_eos,
        opts.encoder_hidden_size,
        opts.word_embedding_size,
        opts.encoder_num_hidden_layers,
        opts.encoder_dropout,
        opts.cell_type,
        opts.beam_width,
        opts.greedy,
        opts.heads,
        opts.on_max_beam_iter,
    )


def train(opts: argparse.Namespace):

    if opts.tiny_preset:
        opts.english_vocab = Path('data/english_vocab_tiny.txt')
        opts.french_vocab = Path('data/french_vocab_tiny.txt')
        opts.train_prefixes = Path('data/train_tiny.txt')
        opts.dev_prefixes = Path('data/dev_tiny.txt')

    torch.manual_seed(opts.seed)
    french_word2id = a2_dataloader.read_stoi_from_file(opts.french_vocab)
    english_word2id = a2_dataloader.read_stoi_from_file(opts.english_vocab)
    with smart_open(opts.train_prefixes, 'rt') as open_file:
        train_prefixes = open_file.read().strip().split('\n')
    train_dataloader = a2_dataloader.HansardDataLoader(
        opts.training_dir, french_word2id, english_word2id, opts.source_lang,
        train_prefixes, batch_size=opts.batch_size, shuffle=True,
        pin_memory=(opts.device.type == 'cuda'),
        num_workers=1,
    )
    del train_prefixes
    with smart_open(opts.dev_prefixes, 'rt') as open_file:
        dev_prefixes = open_file.read().strip().split('\n')
    dev_dataloader = a2_dataloader.HansardDataLoader(
        opts.training_dir, french_word2id, english_word2id, opts.source_lang,
        dev_prefixes, batch_size=opts.batch_size,
        pin_memory=(opts.device.type == 'cuda'),
        num_workers=1,
    )
    del dev_prefixes, french_word2id, english_word2id
    model = init(opts, train_dataloader)
    # students may initialize model parameters in different orders, resulting in
    # different initial settings. If we re-seed here and call reset_parameters(), which
    # resets parameters in fixed order, we should get the same initial random values
    torch.manual_seed(opts.seed)
    model.reset_parameters()
    model.to(opts.device)
    optimizer = torch.optim.Adam(model.parameters())
    best_bleu = 0.
    num_poor = 0
    epoch = 1
    if opts.patience is None:
        max_epochs = opts.epochs
        patience = float('inf')
    else:
        max_epochs = float('inf')
        patience = opts.patience

    ## Training Visualization ##
    if opts.viz_wandb:
        # View at: https://wandb.ai/<opts.viz_wandb>/csc401-w23-a2
        wandb.init(name=f"Train-{type(model.decoder).__name__}",
                   project="csc401-w23-a2",
                   entity=opts.viz_wandb,
                   sync_tensorboard=(opts.viz_tensorboard is not None))
        wandb.config = {
            "max_epochs": max_epochs,
            "patience": patience,
            "batch_size": opts.batch_size,
            "source_vocab_size": train_dataloader.dataset.source_vocab_size,
            "target_vocab_size": train_dataloader.dataset.target_vocab_size
        }
        wandb.watch(model)
    if opts.viz_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(comment="a2_run")  # outputs to ./runs/

    while epoch <= max_epochs and num_poor < patience:
        model.train()
        loss = a2_training_and_testing.train_for_epoch(
            model, train_dataloader, optimizer, opts.device)
        model.eval()
        with torch.no_grad():
            bleu = a2_training_and_testing.compute_average_bleu_over_dataset(
                model, dev_dataloader,
                dev_dataloader.dataset.target_sos,
                dev_dataloader.dataset.target_eos,
                opts.device,
            )
        log_str = f"Epoch {epoch}: loss={loss}, BLEU={bleu}"
        print(log_str)
        if opts.viz_tensorboard:
            writer.add_scalar("Loss/train", loss, epoch)
            writer.add_scalar("BLEU/train", bleu, epoch)
            writer.add_text("", log_str, epoch)
        if opts.viz_wandb:
            #wandb.log({"loss": loss, "global_step": epoch})
            wandb.log({"bleu": bleu, "loss": loss})

        if bleu < best_bleu:
            num_poor += 1
        else:
            num_poor = 0
            best_bleu = bleu
        epoch += 1
    if epoch > max_epochs:
        print(f'Finished {max_epochs} epochs')
    else:
        print(f'BLEU did not improve after {patience} epochs. Done.')

    if opts.viz_tensorboard:
        writer.flush()  # Ensure all pending events have been written to disk
        writer.close()
    if opts.viz_wandb:
        wandb.finish()
    model.cpu()
    with smart_open(opts.model_path, 'wb') as model_file:
        torch.save(model.state_dict(), model_file)

def test(opts: argparse.Namespace):
    french_word2id = a2_dataloader.read_stoi_from_file(opts.french_vocab)
    english_word2id = a2_dataloader.read_stoi_from_file(opts.english_vocab)
    dataloader = a2_dataloader.HansardDataLoader(
        opts.testing_dir, french_word2id, english_word2id, opts.source_lang,
        batch_size=opts.batch_size,
        pin_memory=(opts.device.type == 'cuda')
    )
    del french_word2id, english_word2id
    model = init(opts, dataloader)
    with smart_open(opts.model_path, 'rb') as model_file:
        state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    del state_dict
    model.to(opts.device)
    model.eval()
    with torch.no_grad():
        bleu = a2_training_and_testing.compute_average_bleu_over_dataset(
            model, dataloader,
            dataloader.dataset.target_sos,
            dataloader.dataset.target_eos,
            opts.device,
        )
    print(f'The average BLEU score over the test set was {bleu}')

def interact(opts: argparse.Namespace):
    french_word2id = a2_dataloader.read_stoi_from_file(opts.french_vocab)
    english_word2id = a2_dataloader.read_stoi_from_file(opts.english_vocab)

    dataset = a2_dataloader.HansardEmptyDataset(french_word2id, english_word2id, opts.source_lang)
    dataloader = argparse.Namespace(dataset=dataset)

    model = init(opts, dataloader)
    with smart_open(opts.model_path, 'rb') as model_file:
        state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    del state_dict
    model.to(opts.device)
    model.eval()
    model.dataset = dataset

    print(f'Trained model from path {opts.model_path.name} loaded as the object `model`')

    code.interact(local={'model': model})


def main(args: Optional[List[str]] = None) -> int:
    parser = build_parser()
    opts = parser.parse_args(args)
    if opts.command == 'vocab':
        build_vocab(opts)
    elif opts.command == 'split':
        build_data_train_dev_split(opts)
    elif opts.command == 'train':
        train(opts)
    elif opts.command == 'test':
        test(opts)
    elif opts.command == 'interact':
        interact(opts)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(
        help='Specific commands', dest='command')
    build_vocab_parser(subparsers)
    build_data_train_dev_split_parser(subparsers)
    build_training_parser(subparsers)
    build_testing_parser(subparsers)
    build_interact_parser(subparsers)
    return parser


def build_vocab_parser(
        subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser('vocab', help='Build the vocab file')
    parser.add_argument(
        'lang', choices=['e', 'f'],
        help="What language we're building the vocabulary for"
    )
    parser.add_argument(
        'out',
        type=Path, nargs='?',
        default=sys.stdout,
        help='Where to output the vocab file to. Defaults to stdout. If the '
        'path ends with ".gz", will gzip the file.'
    )
    parser.add_argument(
        '--training-dir', type=Path,
        default=Path('/u/cs401/A2/data/Hansard/Training/'),
        help='Where the training data is located'
    )
    parser.add_argument(
        '--max-vocab', metavar='V', type=lower_bound, default=20000,
        help='The maximum size of the vocabulary. Words with lower frequency '
        'will be cut first'
    )
    parser.add_argument(
        '--min-freq', metavar='F', type=lower_bound, default=1,
        help="The min. frequency of a token to be included in the the vocabulary. Tokens appearing less will be "
             "omitted."
    )
    parser.add_argument(
        '--specials', default=["<s>", "</s>", "<blank>", "<unk>"], nargs='*', type=str,
        help="List of special symbols to include in the vocab at the beginning"
    )

    return parser


def build_data_train_dev_split_parser(
        subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        'split',
        help='Split training data into a training and dev set "randomly". '
        'Places training data prefixes in the first output file and test data '
        'prefixes in the second file.'
    )
    parser.add_argument(
        '--train-prefixes', type=Path,
        default=Path('data/train_prefixes.txt'),
        help='Where to output training data prefixes'
    )
    parser.add_argument(
        '--dev-prefixes', type=Path,
        default=Path('data/dev_prefixes.txt'),
        help='Where to output development data prefixes'
    )
    parser.add_argument(
        '--training-dir', type=Path,
        default=Path('/u/cs401/A2/data/Hansard/Training/'),
        help='Where the training data is located'
    )
    parser.add_argument(
        '--limit', metavar='N', type=lambda v: lower_bound(v, 2), default=None,
        help='Limit on the total number of documents to consider.'
    )
    parser.add_argument(
        '--proportion-training', metavar='(0, 1)', type=proportion,
        default=0.9,
        help='The proportion of total samples that will be used for training'
    )
    parser.add_argument(
        '--seed', metavar='I', type=int, default=0,
        help='The seed used in shuffling'
    )
    return parser


def build_training_parser(
        subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser('train', help='Train an encoder/decoder')
    parser.add_argument(
        'model_path', type=Path,
        help='Where to store the resulting model'
    )
    parser.add_argument(
        '--training-dir', type=Path,
        default=Path('/u/cs401/A2/data/Hansard/Training/'),
        help='Where the training data is located'
    )
    parser.add_argument(
        '--english-vocab', type=Path,
        default=Path('data/english_vocab.txt'),
        help='English vocabulary file'
    )
    parser.add_argument(
        '--french-vocab', type=Path,
        default=Path('data/french_vocab.txt'),
        help='French vocabulary file'
    )
    parser.add_argument(
        '--train-prefixes', type=Path,
        default=Path('data/train_prefixes.txt'),
        help='Where training data prefixes are saved'
    )
    parser.add_argument(
        '--dev-prefixes', type=Path,
        default=Path('data/dev_prefixes.txt'),
        help='Where development data prefixes are saved'
    )
    parser.add_argument(
        '--source-lang', choices=['f', 'e'], default='f',
        help='The source language'
    )
    stopping = parser.add_mutually_exclusive_group()
    stopping.add_argument(
        '--epochs', type=lower_bound, metavar='E', default=5,
        help='The number of epochs to run in total. Mutually exclusive with '
        '--patience. Defaults to 5.'
    )
    stopping.add_argument(
        '--patience', type=lower_bound, metavar='P', default=None,
        help='The number of epochs with no BLEU improvement after which to '
        'call it quits. If unset, will train until the epoch limit instead.'
    )
    parser.add_argument(
        '--batch-size', metavar='B', type=lower_bound, default=128,
        help='The number of sequences to process at once'
    )
    parser.add_argument(
        '--device', metavar='DEV', type=torch.device,
        default=torch.device('cpu'),
        help='Where to do training (e.g. "cpu", "cuda")'
    )
    parser.add_argument(
        '--seed', type=int, metavar='S', default=0,
        help='The random seed, for reproducibility')
    parser.add_argument(
        "--viz-wandb", type=str, default=None, help="Visualize using WandB <username>"
    )
    parser.add_argument(
        "--viz-tensorboard", action="store_true", help="Visualize using Tensorboard"
    )
    parser.add_argument(
        "--tiny-preset", action="store_true",
        help="Run the model with a tiny version of the task. "
        "This flag will overwrite `english_vocab`, `french_vocab`, `train_prefixes` and `dev_prefixes`."
    )
    add_common_model_options(parser)
    return parser


def build_testing_parser(
        subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser('test', help='Evaluate an encoder/decoder')
    parser.add_argument(
        'model_path', type=Path,
        help='Where the model was stored after training. Model parameters '
        'passed via command line should match those from training'
    )
    parser.add_argument(
        '--testing-dir', type=Path,
        default=Path('/u/cs401/A2/data/Hansard/Testing/'),
        help='Where the test data is located'
    )
    parser.add_argument(
        '--english-vocab', type=Path,
        default=Path('data/english_vocab.txt'),
        help='English vocabulary file'
    )
    parser.add_argument(
        '--french-vocab', type=Path,
        default=Path('data/french_vocab.txt'),
        help='French vocabulary file'
    )
    parser.add_argument(
        '--source-lang', choices=['f', 'e'], default='f',
        help='The source language'
    )
    parser.add_argument(
        '--batch-size', metavar='B', type=lower_bound, default=100,
        help='The number of sequences to process at once'
    )
    parser.add_argument(
        '--device', metavar='DEV', type=torch.device,
        default=torch.device('cpu'),
        help='Where to do training (e.g. "cpu", "cuda")'
    )
    add_common_model_options(parser)
    return parser


def build_interact_parser(
        subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser('interact', help='Load and interact with an encoder/decoder')
    parser.add_argument(
        'model_path', type=Path,
        help='Where the model was stored after training. Model parameters '
        'passed via command line should match those from training'
    )
    parser.add_argument(
        '--english-vocab', type=Path,
        default=Path('data/english_vocab.txt'),
        help='English vocabulary file'
    )
    parser.add_argument(
        '--french-vocab', type=Path,
        default=Path('data/french_vocab.txt'),
        help='French vocabulary file'
    )
    parser.add_argument(
        '--source-lang', choices=['f', 'e'], default='f',
        help='The source language'
    )
    parser.add_argument(
        '--device', metavar='DEV', type=torch.device,
        default=torch.device('cpu'),
        help='Where to do training (e.g. "cpu", "cuda")'
    )
    add_common_model_options(parser)
    return parser


def add_common_model_options(parser: argparse.ArgumentParser):
    attn_group = parser.add_mutually_exclusive_group()
    attn_group.add_argument(
        '--with-attention', action='store_true', default=False,
        help='When set, use attention'
    )
    attn_group.add_argument(
        '--with-multihead-attention', action='store_true', default=False,
        help='When set, use multi-head attention'
    )
    attn_group.add_argument(
        '--with-transformer', action='store_true', default=False,
        help='When set, use transformer model architecture'
    )
    parser.add_argument(
        '--heads', metavar='N', default=4, type=int,
        help='The number of heads to use for the multi-head attention mechanism'
    )
    parser.add_argument(
        '--word-embedding-size', metavar='W', type=lower_bound, default=512,
        help='The size of word embeddings in both the encoder and decoder'
    )
    parser.add_argument(
        '--encoder-hidden-size', metavar='H', type=lower_bound, default=512,
        help='The hidden state size in one direction of the encoder'
    )
    parser.add_argument(
        '--encoder-num-hidden-layers', metavar='L', type=lower_bound,
        default=2,
        help='The number of hidden layers in the encoder'
    )
    parser.add_argument(
        '--cell-type', choices=['lstm', 'gru', 'rnn'], default='lstm',
        help='What recurrent architecture to use in both the encoder and '
        'decoder'
    )
    parser.add_argument(
        '--encoder-dropout', metavar='p', type=proportion, default=0.1,
        help='The probability of dropping an encoder hidden state during '
        'training'
    )
    parser.add_argument(
        '--beam-width', metavar='K', type=lower_bound, default=4,
        help='The total number of paths to consider at one time during beam '
        'search'
    )
    parser.add_argument(
        '--greedy', action='store_true',
        help='Use the greedy algorithm instead of beam search for the decoder'
    )
    parser.add_argument(
        '--on-max-beam-iter', choices=['halt', 'raise', 'ignore'], default='halt',
        help='The action to take when reaching the maximum iterations of beam '
        'search. `raise` will raise an exception, `halt` will throw a warning '
        'and halt the search process, and `ignore` will ignore the maximum '
        'iteration limit and continue the search.'
    )

def smart_open(
        path: str, mode: str = 'r') -> Union[gzip.GzipFile, TextIO]:
    if path.suffix == '.gz':
        open_ = gzip.open
        if mode[-1] != 'b':
            mode += 't'
    else:
        open_ = open
    try:
        f = open_(path, mode=mode)
    except OSError as e:
        raise argparse.ArgumentTypeError(
            f"can't open '{path}': {e}")
    return f

def lower_bound(v: str, low: int = 1) -> int:
    v = int(v)
    if v < low:
        raise argparse.ArgumentTypeError(f'{v} must be at least {low}')
    return v

def proportion(v: str, inclusive: bool = False) -> float:
    v = float(v)
    if inclusive:
        if v < 0. or v > 1.:
            raise argparse.ArgumentTypeError(f'{v} must be between [0, 1]')
    else:
        if v <= 0 or v >= 1:
            raise argparse.ArgumentTypeError(f'{v} must be between (0, 1)')
    return v


if __name__ == '__main__':
    sys.exit(main())
