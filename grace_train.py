import argparse
import os.path
import time
from collate.collator import *

from transformers import AutoModel, AutoTokenizer, AutoConfig
from models.grace import GRACE
from preprocess import *
from models.model import DocREModel
from torch.utils.data import *
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from tqdm import tqdm
from augmentation_graph import augmentation
from logger import logger
from matplotlib import pyplot as plt

def set_seed(seeder):
    random.seed(seeder)
    np.random.seed(seeder)
    torch.manual_seed(seeder)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seeder)

def grace_train(args, model, features):
    
    train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    train_iterator = range(int(args.num_train_epochs))
    total_steps = int(len(train_dataloader) * args.num_train_epochs // args.gradient_accumulation_steps)    
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    bert_layer = ['bert_model']
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in bert_layer)], "lr": args.bert_lr},
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in bert_layer)]},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, weight_decay= args.weight_decay)

    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    total_loss = 0
    num_steps = 0
    log_step = 50
    losses = []
    
    model.zero_grad()
    for epoch in tqdm(train_iterator):
        start_time = time.time()
        model.zero_grad()

        for step, batch in enumerate(train_dataloader):
            model.train()
            (
                input_ids, 
                input_mask,
                batch_entity_pos, 
                batch_sent_pos, 
                graph, 
                num_mention, 
                num_entity, 
                num_sent, 
                labels, 
                hts
            ) = batch
            
            graph_first, features_first = augmentation(graph, input_ids, args.feature_prob_first, args.edge_prob_first)
            graph_second, features_second = augmentation(graph, input_ids, args.feature_prob_second, args.edge_prob_second)

            outputs_first = model(
                features_first.to(args.device), 
                input_mask.to(args.device), 
                batch_entity_pos, 
                batch_sent_pos, 
                graph_first.to(args.device), 
                num_mention, 
                num_entity, 
                num_sent)
            
            outputs_second = model(
                features_second.to(args.device), 
                input_mask.to(args.device), 
                batch_entity_pos, 
                batch_sent_pos, 
                graph_second.to(args.device), 
                num_mention, 
                num_entity, 
                num_sent)
            
            loss = model.grace_loss(outputs_first, outputs_second)
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

                    logger.info(
                        '| epoch {:2d} | step {:4d} | min/b {:5.2f} | lr {} | train loss {:5.3f}'.format(
                            epoch, num_steps, elapsed / 60, scheduler.get_lr(), cur_loss))

                    losses.append(cur_loss)

                    total_loss = 0
                    start_time = time.time()
    
    plt.plot(losses)
    plt.savefig(args.grace_loss_viz)
    logger.info(f'Save grace checkpoint.')

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default='./dataset/cdr', type=str)
    parser.add_argument("--transformer_type", default='bert', type=str)
    parser.add_argument("--model_name", default='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext', type=str)
    parser.add_argument("--train_file", default='train_filter.data', type=str)
    parser.add_argument("--dev_file", default='dev_filter.data', type=str)
    parser.add_argument("--test_file", default='test_filter.data', type=str)
    

    # parser.add_argument("--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name")

    # parser.add_argument("--tokenizer_name", default="", type=str, help="Pretrained tokenizer name or path if not the same as model_name")

    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--train_batch_size", default=1, type=int,
                        help="Batch size for training.")

    parser.add_argument("--test_batch_size", default=8, type=int,
                        help="Batch size for testing.")

    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument("--num_labels", default=2, type=int, help="Max number of labels in the prediction.")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float, help="Warm up ratio for Adam.")

    parser.add_argument("--num_train_epochs", default=30.0, type=float, help="Total number of training epochs to perform.")
    # parser.add_argument("--evaluation_steps", default=-1, type=int, help="Number of training steps between evaluations.")
    parser.add_argument("--seed", type=int, default=111, help="random seed for initialization.")
    parser.add_argument("--num_class", default=2, type=int, help="Number of relation types in collate.")

    parser.add_argument("--log_dir", type=str, default='', help="log.")
    
    parser.add_argument("--bert_hidden_dim", default=768, type=int)
    parser.add_argument("--bert_lr", default=5e-5, type=float, help="The initial learning rate for Adam.")
    
    # parser.add_argument("--max_height", type=int, default=42, help="log.")

    parser.add_argument("--gnn_num_layer", type=int, default=2)
    parser.add_argument("--gnn_node_type_embedding", type=int, default=50)
    parser.add_argument("--gnn_hidden_feat_dim", type=int, default=256)
    parser.add_argument("--gnn_output_dim",  type=int, default=128)

    parser.add_argument("--grace_projection_hidden_feat_dim", type=int, default=256)
    parser.add_argument("--grace_projection_out_feat_dim", type=int, default=128)
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
    reader   = read_cdr if "cdr" in args.data_dir else read_gda

    # config collate path
    train_file = os.path.join(args.data_dir, args.train_file)
    dev_file = os.path.join(args.data_dir, args.dev_file)
    test_file = os.path.join(args.data_dir, args.test_file)
    
    train_features = reader(file_in=train_file, save_file="train.cache", tokenizer=tokenizer, max_seq_length=args.max_seq_length)
    dev_features = reader(file_in=dev_file, save_file="dev.cache", tokenizer=tokenizer, max_seq_length=args.max_seq_length)
    # test_features = reader(file_in=test_file, save_file="test.cache",tokenizer=tokenizer, max_seq_length=args.max_seq_length)
    
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

    config = args
    config.bert_config = bert_config
    set_seed(args.seed)
    model = GRACE(config, bert_model)
    model.to(device)
    train_features.extend(dev_features)
    grace_train(config, model, train_features)
    
    torch.save(model.state_dict(), args.save_path)

if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()