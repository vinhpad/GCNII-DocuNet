import argparse
import os.path
import time
from collate.collator import *
from logger import logger
from transformers import AutoModel, AutoTokenizer, AutoConfig
from preprocess import *
from metadata import *
from models.model import DocREModel
from config.run_config import RunConfig
from torch.utils.data import *
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from early_stopping import EarlyStopping
def train(args, model, train_features, dev_features, test_features):
    def finetune(features, optimizer, num_epoch, num_steps):
        best_score = -1
        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn,
                                      drop_last=True)


        train_iterator = range(int(num_epoch))
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        log_step = 50
        total_loss = 0
        valid_losses = []
        early_stopping = EarlyStopping(5)
        for _ in tqdm(train_iterator):
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

                inputs = {'input_ids': input_ids.to(args.device),
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
                valid_losses.append(loss.item())
                loss.backward()
                total_loss += loss.item()


                if step % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    num_steps += 1
                    if num_steps % log_step == 0:
                        cur_loss = total_loss / log_step
                        elapsed = time.time() - start_time
                        total_loss = 0
                        start_time = time.time()

            if args.early_stop:
                valid_loss = np.average(valid_losses)
                early_stopping(valid_loss, model)
                if early_stopping.early_stop:
                    early_stopping.save_checkpoint(valid_loss, model)
                    args.load_path = './checkpoint.pt'
                    break
                
        return best_score
    
    extract_layer = ["extractor", "bilinear"]
    bert_layer = ['bert_model']
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in bert_layer)], "lr": args.bert_lr},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in extract_layer)], "lr": 1e-4},
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in extract_layer + bert_layer)]},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    num_steps = 0   
    set_seed(args)
    model.zero_grad()
    best_score = finetune(train_features, optimizer, args.num_train_epochs, num_steps)
    print(best_score)
    return best_score


def evaluate(args, model, features, tag='test'):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)

    
    preds, golds = [], []
    for batch in dataloader:
        model.eval()
        (
            input_ids, input_mask,
            batch_entity_pos, batch_sent_pos, batch_virtual_pos,
            graph, num_mention, num_entity, num_sent, num_virtual,
            labels, hts
        ) = batch

        inputs = {'input_ids': input_ids.to(args.device),
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
            golds.append(np.concatenate([np.array(label, np.float32) for label in labels], axis=0))
         

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    golds = np.concatenate(golds, axis=0).astype(np.float32)
    tp = ((preds[:, 1] == 1) & (golds[:, 1] == 1)).astype(np.float32).sum()
    tn = ((golds[:, 1] == 1) & (preds[:, 1] != 1)).astype(np.float32).sum()
    fp = ((preds[:, 1] == 1) & (golds[:, 1] != 1)).astype(np.float32).sum()
    precision = tp / (tp + fp + 1e-5)
    recall = tp / (tp + tn + 1e-5)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    output = {
        tag + "_F1": f1 * 100,
        tag + "_P": precision * 100,
        tag + "_R": recall * 100
    }
    return f1, output


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default='./dataset/cdr', type=str)
    parser.add_argument("--transformer_type", default=TRANSFORMER_TYPE, type=str)
    parser.add_argument("--model_name", default=MODEL_NAME, type=str)

    parser.add_argument("--train_file", default=TRAIN_FILE, type=str)
    parser.add_argument("--dev_file", default=DEV_FILE, type=str)
    parser.add_argument("--test_file", default=TEST_FILE, type=str)
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
    parser.add_argument("--evaluation_steps", default=-1, type=int,
                        help="Number of training steps between evaluations.")
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

    parser.add_argument('--save_path', type=str, default='output')

    parser.add_argument('--early_stop', type=str, default=True)

    args = parser.parse_args()
    # Setup device for pytorch
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.device = device

    print(args)

    # Using SciBert
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_special_tokens({
        'additional_special_tokens': [
            '[ENTITY]',
            '[SENT]',
            '[/ENTITY]',
            '[/SENT]'
        ]
    })

    reader = read_cdr

    # config collate path
    train_file = os.path.join(args.data_dir, args.train_file)
    dev_file = os.path.join(args.data_dir, args.dev_file)
    test_file = os.path.join(args.data_dir, args.test_file)
    train_features = reader(file_in=train_file, tokenizer=tokenizer, max_seq_length=args.max_seq_length)
    dev_features = reader(file_in=dev_file, tokenizer=tokenizer, max_seq_length=args.max_seq_length)
    test_features = reader(file_in=test_file, tokenizer=tokenizer, max_seq_length=args.max_seq_length)

    bert_config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=args.model_name,
        num_labels=args.num_class
    )

    bert_config.cls_token_id = tokenizer.cls_token_id
    bert_config.sep_token_id = tokenizer.sep_token_id
    bert_config.transformer_type = args.transformer_type
    

    bert_model = AutoModel.from_pretrained(
        pretrained_model_name_or_path=args.model_name,
        config=bert_config
    )
    bert_model.resize_token_embeddings(len(tokenizer))
    gnn_config = RunConfig.from_json(args.gnn_config_file)
    gnn_config = gnn_config.model.gnn
    set_seed(args)
    model = DocREModel(bert_config, gnn_config, args, bert_model, num_labels=args.num_labels)
    model.to(device)

    if args.early_stop:
        train(args, model, train_features, dev_features, test_features)
        train_features.extend(dev_features)
        args.early_stop = False

    model.load_state_dict(torch.load('./checkpoint.pt'))
    train(args, model, train_features, dev_features, test_features)
    #dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
    test_score, test_output = evaluate(args, model, test_features, tag="test")
    print(test_output)
if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()