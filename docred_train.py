import os
import random
import time
import json
import torch
import argparse
import numpy as np

from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from models.model import DocREModel
from collate.collator import collate_fn
from evaluation import to_official, official_evaluate
from preprocess import read_docred
from logger import logger, setup_log_path
from tqdm import tqdm

def set_seed(seeder):
    random.seed(seeder)
    np.random.seed(seeder)
    torch.manual_seed(seeder)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seeder)
    
    logger.info(f'System set seeder: {seeder}.')

def train(args, model, train_features, dev_features, test_features):

    def finetune(features, optimizer, num_epoch, num_steps):
        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
        train_iterator = range(int(num_epoch))
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        log_step = 50
        total_loss = 0
        total_altop_loss = 0
        total_grace_loss = 0
        for epoch in tqdm(train_iterator):
            start_time = time.time()
            model.zero_grad()
            for step, batch in tqdm(enumerate(train_dataloader)):
                model.train()
                (
                    input_ids, input_mask,
                    batch_entity_pos, batch_sent_pos, batch_virtual_pos,
                    graph, num_mention, num_entity, num_sent, num_virtual,
                    labels, hts
                ) = batch

                inputs = {
                    'input_ids': input_ids.to(args.device),
                    'attention_mask': input_mask.to(args.device),
                    'entity_pos': batch_entity_pos,
                    'sent_pos': batch_sent_pos,
                    'virtual_pos': batch_virtual_pos,
                    'graph': graph.to(args.device),
                    'num_mention': num_mention,
                    'num_entity': num_entity,
                    'num_sent': num_sent,
                    'num_virtual': num_virtual,
                    'labels': labels,
                    'hts': hts,
                }
                outputs = model(**inputs)
                loss = outputs[0] / args.gradient_accumulation_steps
                altop_loss = outputs[1] / args.gradient_accumulation_steps
                grace_loss = outputs[2] / args.gradient_accumulation_steps
                
                loss.backward()
                total_loss += loss.item()
                total_altop_loss += altop_loss.item()
                total_grace_loss += grace_loss.item()
                
                if step % args.gradient_accumulation_steps == 0:
                    
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    num_steps += 1
                    
                    if num_steps % log_step == 0:
                        cur_loss = total_loss / log_step
                        altop_loss = total_altop_loss / log_step
                        grace_loss = total_grace_loss / log_step

                        elapsed = time.time() - start_time

                        logger.info(
                        '| epoch {:2d} | step {:4d} | min/b {:5.2f} | lr {} | train loss {:5.3f} | altop loss {:5.3f} | grace loss {:5.3f}'.format(
                            epoch, num_steps, elapsed / 60, scheduler.get_lr(), cur_loss, altop_loss, grace_loss))

                        total_loss = 0
                        total_altop_loss = 0
                        total_grace_loss = 0
                        start_time = time.time()

                if (step + 1) == len(train_dataloader) - 1 or (args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):
                    eval_start_time = time.time()
                    # _, dev_output = evaluate(args, model, dev_features, tag="dev")
                    _, test_output = evaluate(args, model, test_features, tag="test")
                    logger.info(
                        '| epoch {:3d} | time: {:5.2f}s | test_output:{}'.format(epoch, time.time() - eval_start_time, test_output))
                
                    # logger.info('| epoch {:3d} | best_f1:{}'.format(epoch, best_score))
                    
            if args.save_path != "":
                torch.save({
                    'epoch': epoch,
                    'checkpoint': model.state_dict(),
                    # 'best_f1': best_score,
                    'optimizer': optimizer.state_dict()
                }, args.save_path
                , _use_new_zipfile_serialization=False)


    extract_layer = ["extractor", "bilinear"]
    bert_layer = ['bert_model']
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in bert_layer)], "lr": args.bert_lr},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in extract_layer)], "lr": 1e-4},
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in extract_layer + bert_layer)]},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    num_steps = 0
    set_seed(args.seed)
    model.zero_grad()
    finetune(train_features, optimizer, args.num_train_epochs, num_steps)


def evaluate(args, model, features, tag="dev"):

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    golds = []
    total_loss = 0
    for i, batch in enumerate(dataloader):
        model.eval()
        (
            input_ids, input_mask,
            batch_entity_pos, batch_sent_pos, batch_virtual_pos,
            graph, num_mention, num_entity, num_sent, num_virtual,
            labels, hts
        ) = batch

        inputs = {
            'input_ids': input_ids.to(args.device),
            'attention_mask': input_mask.to(args.device),
            'entity_pos': batch_entity_pos,
            'sent_pos': batch_sent_pos,
            'virtual_pos': batch_virtual_pos,
            'graph': graph.to(args.device),
            'num_mention': num_mention,
            'num_entity': num_entity,
            'num_sent': num_sent,
            'num_virtual': num_virtual,
            'labels': labels,
            'hts': hts,
        }

        with torch.no_grad():
            output = model(**inputs)
            loss = output[0]
            pred = output[1].cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)
            total_loss += loss.item()

    average_loss = total_loss / (i + 1)
    preds = np.concatenate(preds, axis=0).astype(np.float32)
    ans = to_official(preds, features)
    if len(ans) > 0:
        best_f1, _, best_f1_ign, _, re_p, re_r = official_evaluate(ans, args.data_dir)
    output = {
        tag + "_F1": best_f1 * 100,
        tag + "_F1_ign": best_f1_ign * 100,
        tag + "_re_p": re_p * 100,
        tag + "_re_r": re_r * 100,
        tag + "_average_loss": average_loss
    }
    return best_f1, output


def report(args, model, features):

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    for batch in dataloader:
        model.eval()

        inputs = {
            'input_ids': batch[0].to(args.device),
            'attention_mask': batch[1].to(args.device),
            'entity_pos': batch[3],
            'hts': batch[4],
        }

        with torch.no_grad():
            pred = model(**inputs)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    preds = to_official(preds, features)
    return preds


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default='./dataset/cdr', type=str)
    parser.add_argument("--transformer_type", default='', type=str)
    parser.add_argument("--model_name_or_path", default='', type=str)

    parser.add_argument("--train_file", default='', type=str)
    parser.add_argument("--dev_file", default='', type=str)
    parser.add_argument("--test_file", default='', type=str)
    parser.add_argument("--load_path", default="", type=str)

    parser.add_argument("--gnn_config_file", default="config_file/gnn_config.json", type=str,
                        help="Config gnn model")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")

    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")

    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    
    parser.add_argument("--evaluation_steps", default=-1, type=int)

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")

    parser.add_argument("--test_batch_size", default=8, type=int,
                        help="Batch size for testing.")

    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument("--num_labels", default=1, type=int,
                        help="Max number of labels in the prediction.")

    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float,
                        help="Total number of training epochs to perform.")

    parser.add_argument("--seed", type=int, default=111,
                        help="random seed for initialization.")
    parser.add_argument("--num_class", type=int, default=2,
                        help="Number of relation types in collate.")

    parser.add_argument("--unet_in_dim", type=int, default=3,
                        help="unet_in_dim.")
    parser.add_argument("--unet_out_dim", type=int, default=256,
                        help="unet_out_dim.")
    parser.add_argument("--down_dim", type=int, default=256,
                        help="down_dim.")
    parser.add_argument("--channel_type", type=str, default='context-based',
                        help="unet_out_dim.")
    parser.add_argument("--log_dir", type=str, default='',
                        help="log.")
    parser.add_argument("--bert_lr", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_height", type=int, default=42,
                        help="log.")

    
    parser.add_argument("--gnn_num_layer", type=int, default=4)
    parser.add_argument("--gnn_num_node_type", type=int, default=4)
    parser.add_argument("--gnn_node_type_embedding", type=int, default=50)
    parser.add_argument("--gnn_hidden_feat_dim", type=int, default=256)

    parser.add_argument("--grace_projection_hidden_feat_dim", type=int, default=256)
    
    parser.add_argument("--grace_loss_viz", default="")
    parser.add_argument("--tau", type=float, default=0.7)
    
    parser.add_argument('--save_path', type=str, default='output')
    parser.add_argument('--feature_prob_first', type=float, default=0.1)
    parser.add_argument('--feature_prob_second', type=float, default=0.1)
    parser.add_argument('--edge_prob_first', type=float, default=0.1)
    parser.add_argument('--edge_prob_second', type=float, default=0.1)

    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.device = device
    setup_log_path(args.log_dir)
    set_seed(args.seed)
    # Using SciBert
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.add_special_tokens({
        'additional_special_tokens': [
            '[ENTITY]',
            '[SENT]',
            '[/ENTITY]',
            '[/SENT]'
        ]
    })


    train_file = os.path.join(args.data_dir, args.train_file)
    dev_file = os.path.join(args.data_dir, args.dev_file)
    test_file = os.path.join(args.data_dir, args.test_file)
    train_features = read_docred(file_in=train_file, save_file="", tokenizer=tokenizer, max_seq_length=args.max_seq_length)
    dev_features = read_docred(file_in=dev_file, save_file="", tokenizer=tokenizer, max_seq_length=args.max_seq_length)
    test_features = read_docred(file_in=test_file, save_file="",tokenizer=tokenizer, max_seq_length=args.max_seq_length)

    bert_config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=args.model_name_or_path,
        num_labels=args.num_class
    )

    bert_config.cls_token_id = tokenizer.cls_token_id
    bert_config.sep_token_id = tokenizer.sep_token_id
    bert_config.transformer_type = args.transformer_type
    
    bert_model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=bert_config,
    )
    bert_model.resize_token_embeddings(len(tokenizer))
    args.bert_config = bert_config
    model = DocREModel(args, bert_model, num_labels=args.num_labels)
    model.to(device)
    
    if args.load_path == "":
        train_features.extend(dev_features)
        train(args, model, train_features, dev_features, test_features)
    else:
        model.load_state_dict(torch.load(args.load_path)['checkpoint'])
        T_features = test_features
        pred = report(args, model, T_features)

        with open(f'./submit_result/seed_{args.seed}.json', "w") as fh:
            json.dump(pred, fh)

if __name__ == "__main__":
    main()