from typing import List, Callable

import torch
from torch import nn


from rtg import log, device, get_my_args, register_model, Batch
from rtg.classifier import ClassifierModel, ClassificationExperiment, ClassifierTrainer
from rtg.comet.experiment import CometExperiment, HFField
from rtg.classifier.transformer import ClassifierHead, SentenceCompressor



class HFCometExperiment(CometExperiment):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_id = self.model_args['model_id']
        assert self.model_id.startswith('hf:'), 'only huggingface models are supported'
        self.model_id = self.model_id[3:]
        self.src_field = HFField(self.model_id)

    def pre_process(self, args=None, force=False):
        if self._prepared_flag.exists() and not force:
            log.info(f"Pre-processing already done for {self.work_dir}")
            return
        args = args or self.config.get('prep')
        log.info(f"Pre-processing data for {self.model_id}")

        field_types =  self.config['prep']['fields']
        tgt_field_type = field_types[-1]
        assert tgt_field_type in ('class', 'real'), f'Unknown supported type: {tgt_field_type}; supported: class, real'

        # NOTE:  src vocab should match with pretrained model
        # making tgt vocab from train data
        if force or not self._tgt_field_file.exists():
            # target vocabulary; class names. treat each line as a word
            tgt_corpus = []
            if args.get('train_tgt') and not args.get('train_tgt').startswith('stdin:'):
                tgt_corpus.append(args['train_tgt'])
            if args.get('mono_tgt'):
                tgt_corpus.append(args['mono_tgt'])
            assert tgt_corpus, 'prep.train_tgt (not stdin) or prep.mono_tgt must be defined'
            # NLCodec Class Field
            self.tgt_field = self._make_vocab(
                "tgt", self._tgt_field_file, tgt_field_type, corpus=tgt_corpus, vocab_size=-1
            )
            if tgt_field_type == 'class':
                n_classes = self.config['model_args'].get('tgt_vocab')
                if len(self.tgt_field) != n_classes:
                    log.warning(
                        f'model_args.tgt_vocab={n_classes},'
                        f' but found {len(self.tgt_field)} cls in {tgt_corpus}'
                    )
        self._pre_process_parallel(
            'train_src', 'train_tgt', out_file=self.train_db, args=args, line_check=True
        )
        self._pre_process_parallel(
            'valid_src', 'valid_tgt', out_file=self.valid_file, args=args, line_check=True
        )

        self.persist_state()
        self._prepared_flag.touch()


@register_model()
class HFCometClassifier(ClassifierModel):
    model_type = 'hf-comet-cls'
    experiment_type = HFCometExperiment

    def __init__(
        self,
        encoder: nn.Module,
        model_dim: int,
        n_classes: int,
        compressor: SentenceCompressor,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__(n_classes=n_classes)
        self.encoder = encoder
        self._model_dim = model_dim
        self.compressor = compressor
        # [seq1, seq2, |seq1-seq2|, seq1.seq2]   # 4 * model_dim
        self.classifier_head = ClassifierHead(
            input_dim=4 * self.model_dim, n_classes=n_classes, hid_dim=self.model_dim
        )
        self.freeze_encoder = freeze_encoder

    def init_params(self, scheme='xavier'):
        assert scheme == 'xavier'  # only supported scheme as of now
        # Initialize parameters with xavier uniform
        # exclude pretrained encoder parameters from overwriting
        for module in [self.compressor, self.classifier_head]:
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def get_trainable_params(self, include=None, exclude=None):
        if not include and not exclude or include == 'all':
            return super().get_trainable_params()
        if exclude:
            raise Exception("Exclude not supported yet. Please use include")
            # TODO: implement it later when it is really really needed!
        assert isinstance(include, list)
        # a valid example for include
        # 'src_embed', 'compressor', 'classifier', 'encoder:0,1,2,3,...,n-1',  # encoder:layers
        param_groups = []
        for sub_name in include:
            if hasattr(self, sub_name):
                log.info(f"Trainable parameters <-- {sub_name}")
                param_groups.extend(getattr(self, sub_name).parameters())
            elif sub_name.startswith('encoder:'):  # sub select layers
                sub_name, layers = sub_name.split(':')  # encoder:layer_idx
                n_layer = len(self.encoder.layer)
                layers = [int(x) for x in layers.split(',')]
                # negative indices are supported
                layers = [x if x >= 0 else n_layer + x for x in layers]
                for layer_idx in layers:
                    log.info(f'Trainable parameters <-- {sub_name}[{layer_idx}] ')
                    layer = self.encoder.layer[layer_idx]
                    param_groups.extend(layer.parameters())
            else:
                raise Exception(f'{sub_name} not supported or invalid')
        return param_groups

    @property
    def model_dim(self):
        return self._model_dim

    @property
    def vocab_size(self):
        return self._vocab_size

    def forward(self, seq1, seq2, seq1_mask=None, seq2_mask=None, pad_val=-1, score='logits'):

        if seq1_mask is None and pad_val >= 0:
            seq1_mask = self.get_padding_mask(seq1, pad_val)
        if seq2_mask is None and pad_val >= 0:
            seq2_mask = self.get_padding_mask(seq2, pad_val)

        with torch.set_grad_enabled(not self.freeze_encoder):
            out1 = self.encoder(seq1, seq1_mask)
            out2 = self.encoder(seq2, seq2_mask)

        # compressor attn is from torch.nn; it takes float or 
        seq1_repr = self.compressor(out1.last_hidden_state, seq1_mask.bool())
        seq2_repr = self.compressor(out2.last_hidden_state, seq2_mask.bool())
        combo_repr = self.comet_repr(seq1_repr, seq2_repr)
        return self.classifier_head(combo_repr, score=score)

    def get_padding_mask(self, batch, pad_val:int, mtype='explicit'):
        # some are additive mask (0=keep, -inf=ignore)
        # some are multiplicative mask (1=keep, 0=ignore)
        # some are explicit mask (0=keep, 1=ignore)  <-- here we are using this
        return (batch == pad_val).int()
            

    def comet_repr(self, seq1_repr, seq2_repr):
        # [seq1, seq2, |seq1-seq2|, seq1.seq2]
        return torch.cat(
            [seq1_repr, seq2_repr, torch.abs(seq1_repr - seq2_repr), seq1_repr * seq2_repr], dim=1
        )

    @classmethod
    def make_model(
        cls,
        exp: ClassificationExperiment,
        model_id: str,
        src_vocab: int,
        n_classes: int,
        freeze_encoder=True,
    ) -> ClassifierModel:
        args = get_my_args(exclusions=['exp', 'cls'])
        log.info(f"Creating model {cls.__name__} with args: {args}")
        assert model_id.startswith('hf:'), 'only huggingface hub models are supported'
        model_id = model_id[3:]

        from transformers import AutoModel, M2M100Model

        encoder = AutoModel.from_pretrained(model_id)
        if isinstance(encoder, M2M100Model):
            # this is an encoder-decoder model, we need to extract the encoder
            encoder = encoder.encoder
        model_dim = encoder.config.hidden_size
        assert model_dim % 64 == 0, 'model_dim must be a multiple of 64'
        n_heads = int(model_dim // 64)
        compressor_attn = nn.MultiheadAttention(model_dim, num_heads=n_heads, dropout=0.1, batch_first=True)
        # rtg impl masks positions with false/0, but torch impl masks positions with true/1
        # this multihead attn has masking compatible with torch/transformer
        compressor = SentenceCompressor(model_dim, attn=compressor_attn)
        model = cls(
            encoder,
            model_dim=model_dim,
            n_classes=n_classes,
            compressor=compressor,
            freeze_encoder=freeze_encoder,
        )
        model.init_params()
        return model, args

    @classmethod
    def make_trainer(cls, *args, **kwargs):
        return HFCometTrainer(*args, model_factory=cls.make_model, **kwargs)


class HFCometTrainer(ClassifierTrainer):

    def _batch_step(self, batch: Batch, take_step=False, train_mode=False):
        """Take a single step of training or validation on a batch
        :param batch: batch object
        :param take_step: whether to take optimizer step  (requires train_mode=True). Useful for gradient accumulation.
        :param train_mode: whether to run in train mode i.e., with grads no grads
        """
        scores = self.model(
            seq1=batch.x1s,
            seq2=batch.x2s,
            pad_val=batch.pad_val,
            score=self.criterion.input_type,
        )
        loss = self.loss_func(scores=scores, labels=batch.ys, train_mode=train_mode, take_step=take_step)
        return loss, scores