import os
import torch
import numpy as np
from logger import logger
from torch.utils.data import DataLoader
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from dataset.collator import collate_fn
from tqdm import tqdm


class Trainer:
    def __init__(self, args, model, train_feature, test_feature):

        self.optimizer = None
        self.epoch = 0
        self.max_epoch = int(args.num_train_epochs)
        self.scheduler = None
        self.train_loader = None
        self.model = model
        self.train_feature = train_feature
        self.test_feature = test_feature
        self.args = args
        self.device = args.device
        

    def train(self):
        try:
            self.before_train_loop()
            for self.epoch in range(self.max_epoch):
                self.before_epoch()
                self.train_one_epoch(self.epoch)
                self.after_epoch()
            #self.strip_model()
        except Exception as _:
            logger.error('ERROR in training loop or eval/save model.')
            raise
        finally:
            pass
            #self.strip_model()

    def before_train_loop(self):
        # logger.info(f'Start epoch {self.epoch}')

        new_layer = ["extractor", "bilinear"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in new_layer)], },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in new_layer)], "lr": 1e-4},
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.args.learning_rate,
                          eps=self.args.adam_epsilon)

        self.model.zero_grad()

        self.train_loader = DataLoader(
            self.train_feature,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True
        )

        total_steps = int(len(self.train_loader) * self.max_epoch // self.args.gradient_accumulation_steps)
        warmup_steps = int(total_steps * self.args.warmup_ratio)
        logger.info("Total steps: {}".format(total_steps))
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=warmup_steps,
                                                         num_training_steps=total_steps)

        #logger.info("Warmup steps: {}".format(warmup_steps))

    def before_epoch(self):
        pass

    def train_one_epoch(self, epoch):
        self.model.zero_grad()
        for step, batch in tqdm(enumerate(self.train_loader)):
            self.model.train()
            (
                input_ids, input_mask,
                batch_entity_pos, batch_sent_pos, batch_virtual_pos, 
                graph, num_mention, num_entity, num_sent, num_virtual,
                labels, labels_node, hts
            ) = batch

            inputs = {'input_ids': input_ids.to(self.device),
                      'attention_mask': input_mask.to(self.device),
                      'entity_pos': batch_entity_pos,
                      'sent_pos': batch_sent_pos,
                      'virtual_pos': batch_virtual_pos,
                      'graph': graph.to(self.device),
                      'num_mention': num_mention,
                      'num_entity': num_entity,
                      'num_sent': num_sent,
                      'num_virtual': num_virtual,
                      'labels': labels,
                      'labels_node': labels_node,
                      'hts': hts,
                      }

            outputs = self.model(**inputs)
            loss = outputs[0] / self.args.gradient_accumulation_steps
            loss.backward()
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            if step % self.args.gradient_accumulation_steps == 0:
                # if args.max_grad_norm > 0:
                #     torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()
                #num_steps += 1
            # wandb.log({"loss": loss.item()}, step=num_steps)
            # if step % 100 == 0:
            #    logger.info(loss)

    def strip_model(self):
        torch.save(self.model.state_dict(), os.path.join(self.args.save_path, 'model.pt'))


    def after_epoch(self):
        pass
        # self.evaluate()
        # logger.info(f'End epoch {self.epoch}')

    def evaluate(self):
        test_loader = DataLoader(
            self.test_feature,
            batch_size=self.args.test_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False
        )

        preds, golds = [], []
        for batch in test_loader:
            self.model.eval()
            (
                input_ids, input_mask,
                entity_pos, sent_pos, virtual_pos,
                graph, num_mention, num_entity, num_sent, num_virtual,
                labels, labels_node, hts
            ) = batch

            inputs = {'input_ids': input_ids.to(self.device),
                      'attention_mask': input_mask.to(self.device),
                      'entity_pos': entity_pos,
                      'sent_pos': sent_pos,
                      'virtual_pos':virtual_pos,
                      'graph': graph.to(self.device),
                      'num_mention': num_mention,
                      'num_entity': num_entity,
                      'num_sent': num_sent,
                      'num_virtual': num_virtual,
                      'hts': hts,
                      }

            with torch.no_grad():
                pred, *_ = self.model(**inputs)
                pred = pred.cpu().numpy()
                pred[np.isnan(pred)] = 0
                preds.append(pred)
                golds.append(np.concatenate([np.array(label, np.float32) for label in labels], axis=0))
        # print(preds)
        preds = np.concatenate(preds, axis=0).astype(np.float32)
        golds = np.concatenate(golds, axis=0).astype(np.float32)

        tp = ((preds[:, 1] == 1) & (golds[:, 1] == 1)).astype(np.float32).sum()
        tn = ((golds[:, 1] == 1) & (preds[:, 1] != 1)).astype(np.float32).sum()
        fp = ((preds[:, 1] == 1) & (golds[:, 1] != 1)).astype(np.float32).sum()
        precision = tp / (tp + fp + 1e-5)
        recall = tp / (tp + tn + 1e-5)
        f1 = 2 * precision * recall / (precision + recall + 1e-5)
        logger.info(f1)
