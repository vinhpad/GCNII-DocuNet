import argparse
import os.path
import torch
from logger import logger
from transformers import AutoModel, AutoTokenizer, AutoConfig
from dataset.utils import read_cdr
from metadata import *
from models.model import GCN
from config.run_config import RunConfig
from core.engine import Trainer
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default=DATA_DIR, type=str)
    parser.add_argument("--transformer_type", default=TRANSFORMER_TYPE, type=str)
    parser.add_argument("--model_name", default=MODEL_NAME, type=str)

    parser.add_argument("--train_file", default=TRAIN_FILE, type=str)
    parser.add_argument("--dev_file", default=DEV_FILE, type=str)
    parser.add_argument("--test_file", default=TEST_FILE, type=str)
    parser.add_argument("--load_path", default="", type=str)

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

    parser.add_argument("--learning_rate", default=1e-6, type=float,
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
    parser.add_argument("--seed", type=int, default=66,
                        help="random seed for initialization.")
    parser.add_argument("--num_class", type=int, default=2,
                        help="Number of relation types in dataset.")
    parser.add_argument('--config_path', type=str, default='config_file/cdr_config.json')
    parser.add_argument('--save_path', type=str, default='output')
    args = parser.parse_args()

    # Setup device for pytorch
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    logger.info(f'Using device: {device}')

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
    bert_config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=args.model_name,
        num_labels=args.num_class
    )
    bert_model = AutoModel.from_pretrained(
        pretrained_model_name_or_path=args.model_name,
        config=bert_config
    )
    bert_model.resize_token_embeddings(len(tokenizer))
    reader = read_cdr

    # config dataset path
    train_file = os.path.join(args.data_dir, args.train_file)
    dev_file = os.path.join(args.data_dir, args.dev_file)
    test_file = os.path.join(args.data_dir, args.test_file)

    train_features = reader(file_in=train_file, tokenizer=tokenizer, max_seq_length=args.max_seq_length)
    dev_features = reader(file_in=dev_file, tokenizer=tokenizer, max_seq_length=args.max_seq_length)
    test_feature = reader(file_in=test_file, tokenizer=tokenizer, max_seq_length=args.max_seq_length)

    # print(train_features)
    bert_config.cls_token_id = tokenizer.cls_token_id
    bert_config.sep_token_id = tokenizer.sep_token_id
    bert_config.transformer_type = args.transformer_type

    config_path = args.config_path
    config = RunConfig.from_json(config_path)
    model = GCN(config.model, bert_model, device)
    model.to(device)

    args.device = device
    #experiment_dir = setup_experiment_dir(config, tokenizer, bert_model)
    #logger = get_logger(os.path.join(experiment_dir, 'log.txt'))
    # model.to(device)

    train_features.extend(dev_features)
    trainer = Trainer(args, model, train_features, test_feature)
    trainer.train()
    trainer.evaluate()
