import time
from typing import Callable, List, Optional, Tuple, Union
from pathlib import Path
import pickle
import torch

import torch.nn.functional as F
import tqdm
from torch.cuda.amp import autocast

from rtg import (
    EarlyStopper,
    SteppedTrainer,
    TrainerState,
    device,
    dtorch,
    log,
)
from rtg.eval.clsmetric import ClsMetric

from . import ClassifierModel, ClassificationExperiment


class ClassifierTrainer(SteppedTrainer):
    def __init__(
        self,
        exp: ClassificationExperiment,
        model: Optional[ClassifierModel] = None,
        model_factory: Optional[Callable] = None,
    ):
        super().__init__(exp, model, model_factory=model_factory)
        self.exp: ClassificationExperiment = exp
        assert isinstance(
            self.core_model, ClassifierModel
        ), f'Expected an instance of ClassifierModel; but found {type(self.core_model)}'
        chunk_size = self.init_args.get('chunk_size', -1)
        if chunk_size > 0:
            log.warning("chunk_size not supported for this setup; it is ignored")

        if self.n_gpus > 1:  # Multi GPU mode
            raise Exception(
                f"Please use: python -m rtg.distrib.launch -G {self.n_gpus} \n "
                f" or set single GPU by: export CUDA_VISIBLE_DEVICES=0 "
            )

        self.classifier_head = self.core_model.classifier_head
        # required for validation metrics
        assert self.criterion.input_type in (
            'logits',
            'probs',
            'softmax',
            'log_probs',
            'log_softmax',
        ), f'Expected probs or log_probs, but got {self.criterion.input_type}'
    
        self.sync_dir: Path = self.exp.work_dir / 'sync-dir'
        if dtorch.is_distributed and dtorch.is_global_main:
            self.sync_dir.mkdir(exist_ok=True, parents=True)


    def loss_func(self, scores, labels, train_mode=False, take_step=False):
        loss = self.criterion(scores, labels, normalizer=len(labels), mask_out=None)
        if train_mode:  # don't do this for validation set
            dtorch.backward(loss)
            if take_step:
                dtorch.step(self.opt)
        result = loss.item()
        return result

    def run_valid_epoch(self, val_data):
        """
        :param data_iter: data iterator
        :return: loss value
        """
        start = time.time()
        total_loss = 0.0
        num_batches = 0
        model = self.core_model
        assert not model.training
        label_ids, pred_ids, pred_probs = [], [], []
        data_bar = tqdm.tqdm(val_data, unit='batch', dynamic_ncols=True, mininterval=2, smoothing=0.1)
        for i, batch in enumerate(data_bar):
            with autocast(enabled=dtorch.fp16):
                if self.n_gpus <= 1:  # if not dataparallel, then move
                    batch = batch.to(device)
                loss, scores = self._batch_step(batch, take_step=False, train_mode=False)

                total_loss += loss
                num_batches += 1
                elapsed = time.time() - start
                data_bar.set_postfix_str(f'Loss:{loss:.4f}, {int(len(batch) / elapsed)}item/s', refresh=False)

                label_ids += batch.ys.tolist()
                if self.criterion.input_type == 'logits':
                    # softmax was not applied in batch_step. Apply here
                    probs = F.softmax(scores, dim=1)
                elif self.criterion.input_type in ('log_probs', 'log_softmax'):
                    probs = scores.exp()
                else:  # scores are already normalized
                    assert self.criterion.input_type in ('probs', 'softmax')
                    probs = scores
                if self.exp.tgt_field.scheme == 'class':
                    if self.model.n_classes == 1:  # binary classification
                        threshold = 0.5
                        probs = probs.squeeze(1)
                        pred_probs += probs.tolist()
                        pred_ids += (probs >= threshold).long().tolist()
                    else:
                        top1_probs, top1_idx = probs.max(dim=1)
                        pred_ids += top1_idx.tolist()
                        pred_probs += top1_probs.tolist()
            start = time.time()
        data_bar.close()
        assert num_batches > 0, "No batches found. Please check the val data loader"
        loss_avg = total_loss / num_batches
        metrics_dict = dict(loss=loss_avg)

        if self.exp.tgt_field.scheme == 'class':
            cls_metrics = self.validate_cls_metrics(label_ids, pred_ids, pred_probs)
            metrics_dict.update(cls_metrics)

        self.tbd.add_scalars('val_performance', metrics_dict, self.opt.curr_step)
        if not self.exp.read_only:
            path = self.exp.model_dir / 'validation.metrics.tsv'
            write_header = not path.exists()
            with path.open('a') as out:
                if write_header:
                    header = ['step'] + list(metrics_dict.keys())
                    out.write('\t'.join(header) + '\n')
                rec = [self.opt.curr_step] + list(metrics_dict.values())
                out.write('\t'.join(f'{v:g}' for v in rec) + '\n')
        return loss_avg, metrics_dict

    def run_valid_epoch_ddp(self, val_data):
        """
        Validation that uses all workers in DDP mode.  
        Strong assumption: validation data is presented in the same order for all workers in DDP
        :param data_iter: data iterator
        :return: loss value
        """
        start = time.time()
        total_loss = 0.0
        num_batches = 0
        model = self.core_model
        assert not model.training
        label_ids, pred_ids, pred_probs = [], [], []
        pbar_disable = not dtorch.is_local_main
        data_bar = tqdm.tqdm(val_data, unit='batch', dynamic_ncols=True, mininterval=2, smoothing=0.1, disable=pbar_disable)
        for i, batch in enumerate(data_bar):
            with autocast(enabled=dtorch.fp16):
                if dtorch.is_distributed:
                    if (i % dtorch.world_size != dtorch.global_rank):
                        continue   # skip 
                    else:
                        pass   # belongs to this rank; continue
                else:
                    # if not dataparallel, then move tensors to the desired device
                    batch = batch.to(device)

                loss, scores = self._batch_step(batch, take_step=False, train_mode=False)
                total_loss += loss
                num_batches += 1
                elapsed = time.time() - start
                data_bar.set_postfix_str(f'Loss:{loss:.4f}, {int(len(batch) / elapsed)}item/s', refresh=False)

                label_ids += batch.ys.tolist()
                if self.criterion.input_type == 'logits':
                    # softmax was not applied in batch_step. Apply here
                    probs = F.softmax(scores, dim=1)
                elif self.criterion.input_type in ('log_probs', 'log_softmax'):
                    probs = scores.exp()
                else:  # scores are already normalized
                    assert self.criterion.input_type in ('probs', 'softmax')
                    probs = scores
                if self.exp.tgt_field.scheme == 'class':
                    if self.model.n_classes == 1:  # binary classification
                        threshold = 0.5
                        probs = probs.squeeze(1)
                        pred_probs += probs.tolist()
                        pred_ids += (probs >= threshold).long().tolist()
                    else:
                        top1_probs, top1_idx = probs.max(dim=1)
                        pred_ids += top1_idx.tolist()
                        pred_probs += top1_probs.tolist()
            start = time.time()
        data_bar.close()

        if dtorch.is_distributed:
            # we are syncing states using file system::  write -> barrier -> read
            if dtorch.global_rank != dtorch.local_rank:
                log.warning("Looks like you have multi node training. Expected shared file system. Otherwise, you are in trouble.")
            rank = dtorch.global_rank
            state = {
                'total_loss': total_loss,
                'num_batches': num_batches,
                'label_ids':  label_ids,
                'pred_ids': pred_ids,
                'pred_probs': pred_probs
            }
            out_path = self.sync_dir / f'validation.{rank}.pkl'
            with out_path.open('wb') as f:
                pickle.dump(state, f)

            dtorch.barrier()
            if not dtorch.is_global_main:
                # we only need result from main
                return None, None
            cache_paths: List[Path] = []
            for _rank in range(dtorch.world_size):
                if rank == _rank:
                    continue
                state_file = self.sync_dir / f'validation.{_rank}.pkl'
                assert state_file.exists()
                with state_file.open('rb') as f:
                    state = pickle.load(f)
                total_loss += state['total_loss']
                num_batches += state['num_batches']
                label_ids.extend(state['label_ids'])
                pred_ids.extend(state['pred_ids'])
                pred_probs.extend(state['pred_probs'])
                cache_paths.append(state_file)

            # delete files, to cleanup for next iteration
            log.info(f'Synchronized validation state. Clearing files: {cache_paths}')
            for p in cache_paths:
                p.unlink()

        assert num_batches > 0, "No batches found. Please check the val data loader"
        loss_avg = total_loss / num_batches
        metrics_dict = dict(loss=loss_avg)

        if self.exp.tgt_field.scheme == 'class':
            cls_metrics = self.validate_cls_metrics(label_ids, pred_ids, pred_probs)
            metrics_dict.update(cls_metrics)

        self.tbd.add_scalars('val_performance', metrics_dict, self.opt.curr_step)
        if not self.exp.read_only:
            path = self.exp.model_dir / 'validation.metrics.tsv'
            write_header = not path.exists()
            with path.open('a') as out:
                if write_header:
                    header = ['step'] + list(metrics_dict.keys())
                    out.write('\t'.join(header) + '\n')
                rec = [self.opt.curr_step] + list(metrics_dict.values())
                out.write('\t'.join(f'{v:g}' for v in rec) + '\n')
        return loss_avg, metrics_dict


    def validate_cls_metrics(self, label_ids, pred_ids, pred_probs):
        class_names = self.exp.tgt_vocab.class_names
        metrics = ClsMetric(prediction=pred_ids, truth=label_ids, clsmap=class_names)
        pred_names = [class_names[pi] for pi in pred_ids]
        res = {
            'accuracy': metrics.accuracy,
            'macrof1': metrics.macro_f1,
            'microf1': metrics.micro_f1,
            'macro_precision': metrics.macro_precision,
            'macro_recall': metrics.macro_recall,
            'maccuracy': metrics.maccuracy,
        }
        step = self.opt.curr_step
        log_conf_mat = len(class_names) < 40
        log.info(f"validation at step={step}\n{metrics.format(confusion=log_conf_mat)}")

        if not self.exp.read_only:
            val_metric_dir = self.exp.work_dir / 'validations'
            val_metric_dir.mkdir(exist_ok=True)
            (val_metric_dir / f'validation-{step:06d}.score.csv').write_text(metrics.format(delim=','))
            with (val_metric_dir / f'validation-{step:06d}.out.tsv').open('w') as out:
                for p_name, p_prob in zip(pred_names, pred_probs):
                    out.write(f'{p_name}\t{p_prob:g}\n')

        return res


    def _batch_step(self, batch, take_step=False, train_mode=False):
        """Take a single step of training or validation on a batch
        :param batch: batch object
        :param take_step: whether to take optimizer step  (requires train_mode=True). Useful for gradient accumulation.
        :param train_mode: whether to run in train mode i.e., with grads no grads
        """
        x_mask = (batch.x_seqs != batch.pad_val).unsqueeze(1)
        scores = self.model(src=batch.x_seqs, src_mask=x_mask, score=self.criterion.input_type)
        loss = self.loss_func(scores=scores, labels=batch.ys, train_mode=train_mode, take_step=take_step)
        return loss, scores

    def train(
        self,
        steps: int,
        check_point: int,
        batch_size: int,
        log_interval=10,
        keep_models=10,
        sort_by='random',
        keep_in_mem=False,
        early_stop=None,
        fine_tune=False,
        **args,
    ):
        """
        :param steps: how many optimizer steps to train (also, means how many batches)
        :param check_point: after how many checkpoints to
        :param batch_size: how many target tokens in batch max ( = max_len * num_sentences)
        :param keep_models: how many checkpts to keep
        :param keep_in_mem: keep training data in memory
        :param early_stop: {patience: N validations, by: loss, enabled: True}
        :param args: any extra args
        :return:
        """

        # Gradient accumulation
        opt_steps = steps
        batches = steps * self.grad_accum_interval
        start_batch = self.start_step * self.grad_accum_interval
        check_point = check_point * self.grad_accum_interval
        if isinstance(batch_size, int):
            max_toks, max_sents = batch_size, float('inf')
        else:
            max_toks, max_sents = batch_size
        if args:
            # no extra args. let user know if an extra arg is passed
            raise Exception(f"Found extra args: {args}")
        log.info(
            f'Going to train for {opt_steps} optimizer steps over {batches} batches'
            f' (from {self.start_step} steps);'
            f' batch_size={batch_size} toks; sort_by={sort_by};'
        )

        if batches <= start_batch:
            raise Exception(
                f'The model was already trained to {self.start_step} steps. '
                f'Please increase the steps or clear the existing models'
            )

        train_data = self.exp.get_train_data(
            batch_size=[max_toks, max_sents],
            steps=batches - start_batch,
            sort_by=sort_by,
            batch_first=True,
            keep_in_mem=keep_in_mem,
            fine_tune=fine_tune,
            y_is_cls=True,
        )

        train_state = TrainerState(self.model, check_point=check_point, unit='item')
        train_state.train_mode(True)
        unsaved_state = False

        batch_count = -1
        stopper = None
        early_stopped_flag = self.exp.model_dir / '_EARLY_STOPPED'
        if early_stopped_flag.exists():
            early_stopped_flag.unlink()
        if early_stop:
            stopper = EarlyStopper(cur_step=self.start_step, **early_stop)

        def _make_checkpoint():
            if dtorch.is_global_main:
                self.make_check_point(keep_models=keep_models)
            
            dtorch.barrier()
            if dtorch.is_distributed:   # all workers load the same state for validation
                last_chkpt_path =  self.exp.last_state_file
                self.load_state(last_chkpt_path)

            with torch.no_grad():
                train_state.train_mode(False)
                val_data = self.exp.get_val_data(
                    batch_size=[max_toks, max_sents],
                    shuffle=False,
                    batch_first=True,
                    sort_desc=False,
                    y_is_cls=True,
                )
                val_loss, val_metrics = self.run_valid_epoch_ddp(val_data)
                return val_loss, val_metrics

        data_bar = tqdm.tqdm(train_data, initial=start_batch, total=batches, unit='batch',
                             dynamic_ncols=True, disable=not dtorch.is_global_main, mininterval=2, smoothing=0.1)
        for batch in data_bar:
            batch_count += 1
            take_step = (batch_count % self.grad_accum_interval) == 0

            with autocast(enabled=dtorch.fp16):
                if self.n_gpus <= 1:  # if not dataparallel, then move
                    batch = batch.to(device)
                loss, _scores = self._batch_step(batch, take_step=take_step, train_mode=True)

            if stopper and take_step:
                stopper.step()
            # Log
            unsaved_state = True
            
            if dtorch.is_global_main and self.opt.curr_step % log_interval == 0:
                self.tbd.add_scalars(
                    'training', {'step_loss': loss, 'learn_rate': self.opt.curr_lr}, self.opt.curr_step
                )

            progress_msg, is_check_pt = train_state.step(len(batch), loss)
            progress_msg += f', LR={self.opt.curr_lr:0.8f}'
            data_bar.set_postfix_str(progress_msg, refresh=False)
            del batch

            # Save checkpoint
            if is_check_pt:
                train_loss = train_state.reset()
                log.info(f"Chkpt Train loss={train_loss:.4g}")
                _, val_metrics = _make_checkpoint()

                train_state.train_mode(True)
                if stopper and dtorch.is_global_main:
                    score = val_metrics.get(stopper.by, None)
                    assert score is not None, f'early stop by {stopper.by} is invalid; try {val_metrics.keys()}'
                    stopper.validation(score)
                    if stopper.is_stop():
                        log.info(
                            f"Stopping at {stopper.cur_step} because {stopper.by}"
                            f" didnt improve over {stopper.patience} checkpoints"
                        )
                        early_stopped_flag.touch()

                dtorch.barrier()
                unsaved_state = False
                if early_stopped_flag.exists():
                    log.info("Main process was early stopped; so stopping this worker process also")
                    break
        data_bar.close()
        # End of training
        if unsaved_state and dtorch.is_global_main:
            _make_checkpoint()

        dtorch.barrier()
        return early_stopped_flag.exists()
