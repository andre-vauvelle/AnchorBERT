import torch.nn as nn
import torch.nn.functional as f
import pytorch_pretrained_bert as Bert
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import AveragePrecision, MetricCollection, AUROC, Precision
from model.bert.components import BertModel
from model.bert.config import BertConfig

import pytorch_lightning as pl


class BERTMLM(Bert.modeling.BertPreTrainedModel, pl.LightningModule):
    """
    For MLM pretraining.
    """

    def __init__(self, token2idx, bert_config, skip_training=False,
                 num_workers=0, token_type='phecode',
                 file_config=None, max_len_seq=256, temperature_scaling=False,
                 verbose=False, tensorboard_name='mlm'):
        config = BertConfig(bert_config['model_config'])
        super(BERTMLM, self).__init__(config)

        self.token2idx = token2idx
        self.n_labels = len(list(self.token2idx.values()))
        self.bert_config = bert_config
        self.model_config = bert_config['model_config']
        self.train_params = bert_config['train_params']
        self.optim_config = bert_config['optim_config']
        self.file_config = file_config
        self.bert_checkpoint_dir = bert_config['bert_checkpoint_dir']

        self.lr = self.optim_config['lr']
        self.batch_size = self.train_params['batch_size']
        self.max_len_seq = max_len_seq
        self.temperature_scaling = temperature_scaling

        self.skip_training = skip_training

        self.symbol_idxs = [token2idx[k] for k in
                            ['PAD', 'MASK', 'CLS', 'UNK']]  # not including SEP as we can use as feature

        self.num_workers = num_workers
        self.token_type = token_type
        feature_dict = bert_config['feature_dict']
        # self.sigmoid = nn.Sigmoid()
        # self.temperature = ModelWithTemperature() if bert_config['temperature'] else None
        self.bert = BertModel(config, feature_dict)
        self.cls = Bert.modeling.BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

        self.loss_func = nn.CrossEntropyLoss(ignore_index=-1)

        self.tb_logger = TensorBoardLogger(save_dir=bert_config['tensorboard_dir'], name=tensorboard_name)

        metrics = MetricCollection(
            [
                AveragePrecision(num_classes=self.n_labels, compute_on_step=False, average='weighted'),
                Precision(compute_on_step=False, average='micro'),
                # AUROC(num_classes=self.n_labels, compute_on_step=False)
            ]
        )

        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def forward(self, batch):
        token_idx, age_idx, position, segment, phecode_idx, mask, mask_labels = batch
        # mask = (-1) * (mask - 1)
        if self.token_type == 'phecode':
            x = phecode_idx
        else:
            x = token_idx
        unpooled_output, _ = self.bert(input_ids=x, age_ids=age_idx, seg_ids=segment, posi_ids=position,
                                       attention_mask=mask,
                                       output_all_encoded_layers=False)
        predictions = self.cls(unpooled_output)
        # predictions = self.sigmoid(logits)
        return predictions, mask_labels

    def training_step(self, batch, batch_idx):
        prediction_logits, mask_labels = self(batch)
        loss = self.loss_func(prediction_logits.view(-1, self.n_labels), mask_labels.view(-1))
        # predictions = f.sigmoid(logits)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        prediction_logits, mask_labels = self(batch)
        loss = self.loss_func(prediction_logits.view(-1, self.n_labels), mask_labels.view(-1))
        self.log('val_loss', loss, prog_bar=True)

        predictions = f.softmax(prediction_logits.view(-1, self.n_labels), dim=1)
        keep = mask_labels.view(-1) != -1
        mask_labels = mask_labels.view(-1)[keep]
        predictions = predictions[keep]
        self.valid_metrics.update(predictions, mask_labels)

    def on_validation_epoch_end(self) -> None:
        output = self.valid_metrics.compute()
        self.valid_metrics.reset()
        self.log_dict(output, prog_bar=True)

    def test_step(self, batch, batch_idx):
        prediction_logits, mask_labels = self(batch)
        loss = self.loss_func(prediction_logits.view(-1, self.n_labels), mask_labels.view(-1))
        self.log('test_loss', loss, prog_bar=True)
        predictions = f.softmax(prediction_logits.view(-1, self.n_labels), dim=1)
        keep = mask_labels.view(-1) != -1
        mask_labels = mask_labels.view(-1)[keep]
        predictions = predictions[keep]
        self.test_metrics.update(predictions, mask_labels)

    def on_test_epoch_end(self):
        output = self.test_metrics.compute()
        self.test_metrics.reset()
        self.log_dict(output, prog_bar=True)

    # def on_validation_epoch_end(self) -> None:
    #     if self.training:
    #         output = self.valid_metrics.compute()
    #         self.valid_metrics.reset()
    #         self.log_dict(output, prog_bar=True)

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        params = self.named_parameters()

        optimizer_grouped_parameters = [
            {'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in params if any(nd in n for nd in no_decay)], 'weight_decay': 0}
        ]

        optimizer = Bert.optimization.BertAdam(optimizer_grouped_parameters,
                                               lr=self.optim_config['lr'],
                                               warmup=self.optim_config['warmup_proportion'])
        return optimizer
