'''
This script handles the training process.
'''
import argparse
import math
import time
import os
import dill as pickle
from tqdm import tqdm
import logging
import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torchtext.legacy.data import Field, Dataset, BucketIterator
from torchtext.legacy.datasets import TranslationDataset

import transformer.Constants as Constants
from transformer.Models import Transformer, LowRankResidualTransformer, AdaptTransformer
from transformer.Optim import ScheduledOptim

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from bleu import idx_to_word


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# "credit goes to Yu-Hsiang, modified by Hongyi Wang"
__author__ = "Yu-Hsiang Huang"


def cal_performance(pred, gold, trg_pad_idx, trg_vocab, batch_size, trg_seq, smoothing=False, is_train=False):
    ''' Apply label smoothing if needed '''

    # seq_logit.view(-1, seq_logit.size(2))
    loss = cal_loss(pred.view(-1, pred.size(2)), gold, trg_pad_idx, smoothing=smoothing)

    # for calculating accuracy
    pred = pred.view(-1, pred.size(2))
    pred = pred.max(1)[1] # seems to return the index of the max elems
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()

    n_word = non_pad_mask.sum().item()

    #return loss, n_correct, n_word, avg_batch_bleu
    return loss, n_correct, n_word


def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
    return loss


def patch_src(src, pad_idx):
    src = src.transpose(0, 1)
    return src


def patch_trg(trg, pad_idx):
    trg = trg.transpose(0, 1)
    trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
    return trg, gold


def train_epoch(model, training_data, optimizer, opt, device, smoothing, scaler=None):
    ''' Epoch operation in training phase'''

    model.train()
    total_loss, n_word_total, n_word_correct = 0, 0, 0 

    desc = '  - (Training)   '
    logger.info("  - (Training)   ")
    global_batch_index = 0
    for batch_index, batch in enumerate(training_data):

        # prepare data
        src_seq = patch_src(batch.src, opt.src_pad_idx).to(device)
        trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch.trg, opt.trg_pad_idx))

        # forward
        optimizer.zero_grad()
        pred = model(src_seq, trg_seq)

        # backward and update parameters
        loss, n_correct, n_word = cal_performance(
            pred, gold, opt.trg_pad_idx, opt.trg_vocab, opt.batch_size, trg_seq,
            smoothing=smoothing, is_train=True)
        
        total_loss += loss.item()
        # gradient accumulation
        loss = loss / opt.gas 
        #loss.backward()
        scaler.scale(loss).backward()

        # we will have to make good use of the last batches in the training loader
        if (batch_index+1) % opt.gas == 0 or batch_index == len(training_data)-1:
            optimizer.step_and_update_lr()
            logger.info("Global batch idx: {} out of {}, local batch idx: {}/{}".format(
                    global_batch_index, int(len(training_data)/opt.gas+1), batch_index, len(training_data)))
            global_batch_index += 1

        # note keeping
        n_word_total += n_word
        n_word_correct += n_correct
        
    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy


def eval_epoch(model, validation_data, device, opt):
    ''' Epoch operation in evaluation phase '''

    model.eval()
    total_loss, n_word_total, n_word_correct = 0, 0, 0

    desc = '  - (Validation) '
    hypotheses_corpus = []
    reference_corpus = []
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc=desc, leave=False):

            # prepare data
            src_seq = patch_src(batch.src, opt.src_pad_idx).to(device)
            trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch.trg, opt.trg_pad_idx))

            # forward
            pred = model(src_seq, trg_seq)

            for seq_index in range(pred.size()[0]):
                seq_pred = pred[seq_index, :, :].max(1)[1]
                sub_trg = trg_seq[seq_index, :]
                # get words
                output_words = idx_to_word(seq_pred, opt.trg_vocab)
                trg_words = idx_to_word(sub_trg, opt.trg_vocab)

                #seq_bleu = sentence_bleu([trg_words.split()], output_words.split())
                #print("@@@@ seq index: {}, hypo: {}, ref: {} seq bleu: {}".format(
                #        seq_index, output_words, trg_words, seq_bleu*100.0))
                hypotheses_corpus.append(output_words.split())
                reference_corpus.append([trg_words.split()])

            loss, n_correct, n_word = cal_performance(
                pred, gold, opt.trg_pad_idx, opt.trg_vocab, opt.batch_size, trg_seq,
                smoothing=False, is_train=False)

            # note keeping
            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss.item()

        system_bleu = corpus_bleu(reference_corpus, hypotheses_corpus) * 100.0
        #print("!!!!!!! System BLEU score: {}, corpus length: {}".format(
        #                            system_bleu, len(reference_corpus)))

    #total_bleu = sum(total_bleu) / len(total_bleu)
    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy, system_bleu


def toggle_residual_layers(lowrank_model, freeze):
    for param_idx, (param_name, param) in enumerate(lowrank_model.named_parameters()):
        if "_res" in param_name:
            param.requires_grad = not freeze


def train(vanilla_model, lowrank_model, training_data, validation_data, test_data, vanilla_optimizer, lowrank_optimizer, device, opt, scaler):
    ''' Start training '''

    log_train_file = os.path.join(opt.output_dir, 'train.log')
    log_valid_file = os.path.join(opt.output_dir, 'valid.log')

    logger.info('[Info] Training performance will be written to file: {} and {}'.format(
        log_train_file, log_valid_file))

    with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
        log_tf.write('epoch,loss,ppl,accuracy\n')
        log_vf.write('epoch,loss,ppl,accuracy\n')


    def print_performances(header, ppl, accu, start_time, lr, bleu=None):
        if bleu is not None:
            print('  - {header:12} ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, bleu: {bleu:.4f}, lr: {lr:8.5f}, '\
                  'elapse: {elapse:3.3f} min'.format(
                      header=f"({header})", ppl=ppl,
                      accu=100*accu, bleu=bleu, elapse=(time.time()-start_time)/60, lr=lr))
        else:
            print('  - {header:12} ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, lr: {lr:8.5f}, '\
                  'elapse: {elapse:3.3f} min'.format(
                      header=f"({header})", ppl=ppl,
                      accu=100*accu, elapse=(time.time()-start_time)/60, lr=lr))

    
    def get_fullrank_epochs(opt):
        '''
        * opt.fr_adjustment_epochs
            * x[INT]: full rank every [INT] epochs
            * "a, b, c, ...": full rank at epochs a, b, c, ...
        '''
        adj_epochs = opt.fr_adjustment_epochs
        if adj_epochs[0] == 'x':
            interval = int(adj_epochs[1:])
            fr_epochs = set(range(opt.fr_warmup_epoch + interval, opt.epoch + 1, interval))
        else:
            fr_epochs = set(int(x.strip()) for x in adj_epochs.split(','))
        return fr_epochs
        
    
    fr_adjustment_epochs = get_fullrank_epochs(opt)
    logger.info(f'* Full-Rank epochs: {sorted(fr_adjustment_epochs)}')
    
    valid_losses = []
    valid_bleu_scores = []

    for epoch_i in range(opt.epoch):
        logger.info('[ Epoch {} ]'.format(epoch_i))

        start = time.time()
        if epoch_i in range(opt.fr_warmup_epoch):  # epoch < warmup_epoch ==> full rank warmup
            logger.info("### Warming up Training, epoch: {}".format(epoch_i))
            train_loss, train_accu = train_epoch(
                vanilla_model, training_data, vanilla_optimizer, opt, device, smoothing=opt.label_smoothing,
                scaler=scaler)

            # Current learning rate
            lr = vanilla_optimizer._optimizer.param_groups[0]['lr']
        elif epoch_i == opt.fr_warmup_epoch:  # epoch == warmup_epoch ==> decompose
            logger.info("### Switching to low-rank Training, epoch: {}".format(epoch_i))


            logger.info("Low rank Transformer: {},  Number of Parameters: {}".format(lowrank_model,
                                            num_params_counter(lowrank_model)))
            decompose_start = torch.cuda.Event(enable_timing=True)
            decompose_end = torch.cuda.Event(enable_timing=True)

            decompose_start.record()
            # est_rank_list = rank_estimation(epoch_i, vanilla_model)
            lowrank_model = decompose_vanilla_model(vanilla_model=vanilla_model, 
                                                    lowrank_model=lowrank_model,
                                                    rank_ratio=0.25)
            decompose_end.record()
            torch.cuda.synchronize()
            decompose_dur = float(decompose_start.elapsed_time(decompose_end))/1000.0
            logger.info("#### Cost for decomposing the weights: {} ....".format(decompose_dur))
            del vanilla_model
            del vanilla_optimizer

            # freeze residual layers
            toggle_residual_layers(lowrank_model, freeze=True)

            train_loss, train_accu = train_epoch(
                lowrank_model, training_data, lowrank_optimizer, opt, device, smoothing=opt.label_smoothing,
                scaler=scaler)
            # Current learning rate
            lr = lowrank_optimizer._optimizer.param_groups[0]['lr']           
        elif epoch_i in fr_adjustment_epochs:  # (epoch > warmup_epoch) and full rank
            logger.info("### Full-Rank Training, epoch: {}".format(epoch_i))

            # unfreeze residual layers
            toggle_residual_layers(lowrank_model, freeze=False)

            train_loss, train_accu = train_epoch(
                lowrank_model, training_data, lowrank_optimizer, opt, device, smoothing=opt.label_smoothing,
                scaler=scaler)
            # Current learning rate
            lr = lowrank_optimizer._optimizer.param_groups[0]['lr']
        else:  # epoch > warmup_epoch and low rank
            logger.info("### Low-Rank Training, epoch: {}".format(epoch_i))

            # freeze residual layers
            toggle_residual_layers(lowrank_model, freeze=True)

            train_loss, train_accu = train_epoch(
                lowrank_model, training_data, lowrank_optimizer, opt, device, smoothing=opt.label_smoothing,
                scaler=scaler)
            # Current learning rate
            lr = lowrank_optimizer._optimizer.param_groups[0]['lr']

        train_ppl = math.exp(min(train_loss, 100))
        print_performances('Training', train_ppl, train_accu, start, lr)

        start = time.time()
        if epoch_i in range(opt.fr_warmup_epoch):
            valid_loss, valid_accu, valid_bleu = eval_epoch(vanilla_model, validation_data, device, opt)
        else:
            valid_loss, valid_accu, valid_bleu = eval_epoch(lowrank_model, validation_data, device, opt)
        valid_ppl = math.exp(min(valid_loss, 100))
        print_performances('Validation', valid_ppl, valid_accu, start, lr, bleu=valid_bleu)

        valid_losses += [valid_loss]
        valid_bleu_scores += [valid_bleu]

        if opt.fr_warmup_epoch >= opt.epoch: # vanilla 
            checkpoint = {'epoch': epoch_i, 'settings': opt, 'model': vanilla_model.state_dict()}
        elif epoch_i < opt.fr_warmup_epoch:
            checkpoint = {'epoch': epoch_i, 'settings': opt, 'model': vanilla_model.state_dict()}
        else:
            checkpoint = {'epoch': epoch_i, 'settings': opt, 'model': lowrank_model.state_dict()}

        if opt.save_mode == 'all':
            model_name = 'model_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
            torch.save(checkpoint, model_name)
        elif opt.save_mode == 'best':
            if opt.fr_warmup_epoch >= opt.epoch: # vanilla
                model_name = 'model_vanilla_seed{}.chkpt'.format(opt.seed)
            else:
                model_name = 'model_pufferfish_seed{}.chkpt'.format(opt.seed)

            #if valid_loss <= min(valid_losses):
            if valid_bleu >= max(valid_bleu_scores):
                torch.save(checkpoint, model_name)
                logger.info('    - [Info] The checkpoint file has been updated.')

                if epoch_i in range(opt.fr_warmup_epoch):
                    test_loss, test_accu, test_bleu = eval_epoch(vanilla_model, test_data, device, opt)
                else:
                    test_loss, test_accu, test_bleu = eval_epoch(lowrank_model, test_data, device, opt)
                test_ppl = math.exp(min(test_loss, 100))
                print("##### On the test set of WMT 16' En-to-De, Test Ppl.: {}, Test BLEU: {:.4f}".format(
                                                test_ppl, test_bleu))

        with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
            log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                epoch=epoch_i, loss=train_loss,
                ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
            log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                epoch=epoch_i, loss=valid_loss,
                ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu))


def decompose_vanilla_model(vanilla_model, lowrank_model, rank_ratio=0.25):
    collected_weights = []
    weight_names = []
    rank_counter = 0
    for p_index, (name, param) in enumerate(vanilla_model.state_dict().items()):
        name = name.split('.')
        if (name[0] in {'encoder', 'decoder'} and
            name[1] == 'layer_stack' and
            name[2] != '0' and
            name[4] != 'layer_norm' and
            name[5] == 'weight'):

            rank = min(param.size()[0], param.size()[1])
            sliced_rank = int(rank * rank_ratio)
            rank_counter += 1

            u, s, v = torch.svd(param)
            u_weight = u * torch.sqrt(s)
            v_weight = torch.sqrt(s) * v
            u_weight_sliced, v_weight_sliced = u_weight[:, 0:sliced_rank], v_weight[:, 0:sliced_rank]
            res_weight = param - torch.matmul(u_weight_sliced, v_weight_sliced.t())

            collected_weights.append(u_weight_sliced)
            weight_names.append(f'{".".join(name)}_u')
            collected_weights.append(v_weight_sliced.t())
            weight_names.append(f'{".".join(name)}_v')
            collected_weights.append(res_weight)
            weight_names.append(f'{".".join(name)}_res')
        else:
            collected_weights.append(param)
            weight_names.append(".".join(name))
                 
    reconstructed_state_dict = {}
    model_counter = 0
    for p_index, (name, param) in enumerate(lowrank_model.state_dict().items()):
        assert param.size() == collected_weights[model_counter].size(), f'{name}: {param.size()}, {collected_weights[model_counter].size()}'
        reconstructed_state_dict[name] = collected_weights[model_counter]
        model_counter += 1
    lowrank_model.load_state_dict(reconstructed_state_dict)
    return lowrank_model


def num_params_counter(model):
    num_elems = 0
    for p_index, (p_name, p) in enumerate(model.named_parameters()):
        num_elems += p.numel()
    return num_elems


def seed(seed):
    # seed = 1234
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #TODO: Do we need deterministic in cudnn ? Double check
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Seeded everything")


def main():
    ''' 
    Usage:
    python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -save_model trained -b 256 -warmup 128000
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument('-data_pkl', default=None)     # all-in-1 data pickle or bpe field

    parser.add_argument('-train_path', default=None)   # bpe encoded data
    parser.add_argument('-val_path', default=None)     # bpe encoded data

    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=2048)
    parser.add_argument('-gas', type=int, default=8)
    #parser.add_argument('-seed', type=int, default=0, 
    #                    help='the random seed to use for the experiment.')

    # parser.add_argument('-d_model', type=int, default=512)
    # parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_model', type=int, default=1024)
    parser.add_argument('-d_inner_hid', type=int, default=4096)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    #parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_head', type=int, default=16)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-warmup','--n_warmup_steps', type=int, default=4000)
    parser.add_argument('-lr_mul', type=float, default=2.0)
    parser.add_argument('-seed', type=int, default=None)

    #parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-dropout', type=float, default=0.3)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')
    parser.add_argument('-scale_emb_or_prj', type=str, default='prj')

    parser.add_argument('-output_dir', type=str, default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')

    parser.add_argument('-fr_warmup_epoch', type=int, default=150)
    parser.add_argument('-fr_adjustment_epochs', type=str, default='x20')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

    # https://pytorch.org/docs/stable/notes/randomness.html
    # For reproducibility
    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        # torch.set_deterministic(True)
        np.random.seed(opt.seed)
        random.seed(opt.seed)

    if not opt.output_dir:
        print('No experiment result will be saved.')
        raise

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
    device = torch.device('cuda' if opt.cuda else 'cpu')

    #========= Loading Dataset =========#

    if all((opt.train_path, opt.val_path)):
        training_data, validation_data = prepare_dataloaders_from_bpe_files(opt, device)
    elif opt.data_pkl:
        training_data, validation_data, test_data = prepare_dataloaders(opt, device)
    else:
        raise

    logger.info(opt)

    vanilla_transformer = Transformer(
        opt.src_vocab_size,
        opt.trg_vocab_size,
        src_pad_idx=opt.src_pad_idx,
        trg_pad_idx=opt.trg_pad_idx,
        trg_emb_prj_weight_sharing=opt.proj_share_weight,
        emb_src_trg_weight_sharing=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout,
        scale_emb_or_prj=opt.scale_emb_or_prj).to(device)

    lowrank_transformer = LowRankResidualTransformer(
        opt.src_vocab_size,
        opt.trg_vocab_size,
        src_pad_idx=opt.src_pad_idx,
        trg_pad_idx=opt.trg_pad_idx,
        trg_emb_prj_weight_sharing=opt.proj_share_weight,
        emb_src_trg_weight_sharing=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout,
        scale_emb_or_prj=opt.scale_emb_or_prj).to(device)
    logger.info("Full rank Transformer: Number of Parameters: {}".format(num_params_counter(vanilla_transformer)))
    logger.info("Low rank Transformer: Number of Parameters: {}".format(num_params_counter(lowrank_transformer)))

    scaler = GradScaler()

    # optimizer
    vanilla_optimizer = ScheduledOptim(
         optim.AdamW(vanilla_transformer.parameters(), 
         betas=(0.9, 0.98), eps=1e-09, weight_decay=5e-4),
         opt.lr_mul, opt.d_model, opt.n_warmup_steps, scaler=scaler)

    lowrank_optimizer = ScheduledOptim(
       optim.Adam(lowrank_transformer.parameters(), betas=(0.9, 0.98), eps=1e-09),
       2.0, opt.d_model, opt.n_warmup_steps, scaler=scaler)

    train(vanilla_transformer, lowrank_transformer, training_data, validation_data, test_data, vanilla_optimizer, lowrank_optimizer, device, opt, scaler)


def prepare_dataloaders_from_bpe_files(opt, device):
    batch_size = opt.batch_size
    MIN_FREQ = 2
    if not opt.embs_share_weight:
        raise

    data = pickle.load(open(opt.data_pkl, 'rb'))
    MAX_LEN = data['settings'].max_len
    field = data['vocab']
    fields = (field, field)

    def filter_examples_with_length(x):
        return len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN

    train = TranslationDataset(
        fields=fields,
        path=opt.train_path, 
        exts=('.src', '.trg'),
        filter_pred=filter_examples_with_length)
    val = TranslationDataset(
        fields=fields,
        path=opt.val_path, 
        exts=('.src', '.trg'),
        filter_pred=filter_examples_with_length)

    opt.max_token_seq_len = MAX_LEN + 2
    opt.src_pad_idx = opt.trg_pad_idx = field.vocab.stoi[Constants.PAD_WORD]
    opt.src_vocab_size = opt.trg_vocab_size = len(field.vocab)

    train_iterator = BucketIterator(train, batch_size=batch_size, device=device, train=True)
    val_iterator = BucketIterator(val, batch_size=batch_size, device=device)
    return train_iterator, val_iterator


def prepare_dataloaders(opt, device):
    batch_size = opt.batch_size
    data = pickle.load(open(opt.data_pkl, 'rb'))

    opt.max_token_seq_len = data['settings'].max_len
    opt.src_pad_idx = data['vocab']['src'].vocab.stoi[Constants.PAD_WORD]
    opt.trg_pad_idx = data['vocab']['trg'].vocab.stoi[Constants.PAD_WORD]

    opt.src_vocab_size = len(data['vocab']['src'].vocab)
    opt.trg_vocab_size = len(data['vocab']['trg'].vocab)

    opt.trg_vocab = data['vocab']['trg'].vocab

    #========= Preparing Model =========#
    if opt.embs_share_weight:
        assert data['vocab']['src'].vocab.stoi == data['vocab']['trg'].vocab.stoi, \
            'To sharing word embedding the src/trg word2idx table shall be the same.'

    fields = {'src': data['vocab']['src'], 'trg':data['vocab']['trg']}

    train = Dataset(examples=data['train'], fields=fields)
    val = Dataset(examples=data['valid'], fields=fields)
    test = Dataset(examples=data['test'], fields=fields)

    train_iterator = BucketIterator(train, batch_size=batch_size, device=device, train=True)
    val_iterator = BucketIterator(val, batch_size=batch_size, device=device)
    test_iterator = BucketIterator(test, batch_size=batch_size, device=device)

    return train_iterator, val_iterator, test_iterator


if __name__ == '__main__':
    main()
