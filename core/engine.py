import time

import logger
from torch.utils.data import DataLoader
from torch.optim import AdamW
class Trainer:
    def __init__(self, args, model, cfg, device):
        self.model = model
        self.args = args
        self.cfg = cfg
        self.device = device

    def finetune(features, optimizer, num_epoch, num_steps):

        train_dataloader = DataLoader(features,
                                      batch_size=args.train_batch_size,
                                      shuffle=True, collate_fn=collate_fn,
                                      drop_last=True)
        train_iterator = range(int(num_epoch))

        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps)
        logger.info("Total steps: {}".format(total_steps))
        logger.info("Warmup steps: {}".format(warmup_steps))
        for epoch in train_iterator:
            model.zero_grad()
            for step, batch in tqdm(enumerate(train_dataloader)):
                model.train()
                (
                    input_ids, input_mask,
                    entity_pos, sent_pos,
                    graph, num_mention, num_entity, num_sent,
                    labels, hts
                ) = batch
                inputs = {'input_ids': input_ids.to(args.device),
                          'attention_mask': input_mask.to(args.device),
                          'entity_pos': entity_pos,
                          'sent_pos': sent_pos,
                          'graph': graph.to(args.device),
                          'num_mention': num_mention,
                          'num_entity': num_entity,
                          'num_sent': num_sent,
                          'labels': labels,
                          'hts': hts,
                          }
                outputs = model(**inputs)
                loss = outputs[0] / args.gradient_accumulation_steps
                loss.backward()

                if step % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    num_steps += 1

                if step % 100 == 0:
                    logger.info(loss)

        os.makedirs(os.path.join(experiment_dir, 'model'))
        torch.save(model.state_dict(), os.path.join(experiment_dir, 'model', 'model.pt'))
        return num_steps

    def train(self):
        try:
            self.before_train_loop()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                self.before_epoch()
                self.train_one_epoch(self.epoch)
                self.after_epoch()
            self.strip_model()
        except Exception as _:
            logger.error('ERROR in training loop or eval/save model.')
            raise
        finally:
            self.train_after_loop()

    def before_train_loop(self):
        print("Training start...")
        new_layer = ["extractor", "bilinear"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in new_layer)], },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in new_layer)], "lr": 1e-4},
        ]

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            eps=self.args.adam_epsilon)

        self.start_spoch = 0
        self.max_epoch = args.e
    def before_epoch(self):
        self.model.zero_grad()

    def train_one_epoch(self, epoch):
        for step, batch