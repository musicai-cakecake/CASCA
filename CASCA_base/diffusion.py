import itertools
from dataclasses import dataclass

import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from tqdm import tqdm
from collections import OrderedDict

import dataloader
import metrics
import models
import noise_schedule
import utils
import numpy as np
import itertools

def _sample_categorical(categorical_probs):
  gumbel_norm = (1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log())
  samples = (categorical_probs / gumbel_norm).argmax(dim=-1)
  return samples

def _unsqueeze(x, reference):
  return x.view(
    * x.shape,
    * ((1,) * (len(reference.shape) - len(x.shape))))

@dataclass
class Loss:
  loss: torch.FloatTensor
  nlls: torch.FloatTensor
  token_mask: torch.FloatTensor


class Diffusion(L.LightningModule):
  def __init__(
    self,
    config,
    tokenizer: transformers.PreTrainedTokenizer):
    super().__init__()
    self.save_hyperparameters()
    self.config = config
    self.tokenizer = tokenizer
    self.vocab_size = self.tokenizer.vocab_size
    self.sampler = self.config.algo.sampler
    self.antithetic_sampling = self.config.training.antithetic_sampling
    self.cross_attn = self.config.algo.cross_attn
    self.ignore_bos = self.config.algo.ignore_bos
    self.mdlm_loss_scale = self.config.algo.mdlm_loss_scale
    if (not hasattr(self.tokenizer, 'mask_token')
        or self.tokenizer.mask_token is None):
      self.mask_index = self.vocab_size
      self.vocab_size += 1
    else:
      self.mask_index = self.tokenizer.mask_token_id
    if hasattr(self.config, 'algo'):
      self.parameterization = self.config.algo.parameterization
    else:
      self.parameterization = self.config.parameterization
    if hasattr(self.config, 'block_size'):
      self.block_size = self.config.block_size
    else:
      self.block_size = self.config.model.length
    if self.parameterization == 'ar':
      self.block_size = 1
    self._validate_xmidi_v2_configuration()
    if self.config.algo.backbone == 'dit':
      self.backbone = models.dit.DIT(
        self.config, vocab_size=self.vocab_size)
    elif self.config.algo.backbone == 'dimamba':
      self.backbone = models.dimamba.DiMamba(
        self.config,
        vocab_size=self.vocab_size,
        pad_token_id=self.tokenizer.pad_token_id)
    elif self.config.algo.backbone == 'hf_dit':
      self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(
        config.eval.checkpoint_path, trust_remote_code=True)
      #  egenerate mask if pretrained model uses flex attention mask
      # and current model uses sdpa mask
      if getattr(self.backbone.config, 'attn_backend', None) == 'flex' and \
        self.config.model.attn_backend == 'sdpa':
        self.backbone.config.attn_backend = 'sdpa'
        for i in self.backbone.backbone.blocks:
          i.attn_backend = 'sdpa'
        self.backbone.backbone.gen_mask(self.config.model.length, self.block_size, attn_backend='sdpa')
    else:
      raise ValueError(f'Unknown backbone: {self.config.algo.backbone}')

    self.T = self.config.algo.T
    self.num_tokens = self.config.model.length

    self.noise = noise_schedule.get_noise(self.config)
    self.metrics = metrics.Metrics(config)

    use_ema = (
      self.config.training.ema > 0
      and not bool(getattr(self.config.eval, 'disable_ema', False)))
    if use_ema:
      self.ema = models.ema.ExponentialMovingAverage(
        self._get_parameters(),
        decay=self.config.training.ema)
    else:
      self.ema = None
    
    self.var_min = self.config.algo.var_min
    if self.var_min:
      self.register_buffer('sampling_eps_min', torch.tensor(
        self.config.training.sampling_eps_min))
      self.register_buffer('sampling_eps_max', torch.tensor(
        self.config.training.sampling_eps_max))
      
    self.time_conditioning = self.config.algo.time_conditioning
    self.neg_infinity = -1000000.0
    self.fast_forward_epochs = None
    self.fast_forward_batches = None
    self._validate_configuration()

  def _validate_xmidi_v2_configuration(self):
    """Fail fast when a V1 geometry or vocabulary reaches the V2 model."""
    data_name = getattr(getattr(self.config, 'data', None), 'train', None)
    if data_name != 'xmidi_getmusic':
      return
    from xmidi_getmusic.constants import (
      BLOCK_SIZE, MASK_ID, PAD_ID, SEQ_LEN, VOCAB_SIZE)
    errors = []
    if int(self.config.model.length) != SEQ_LEN:
      errors.append(
        f'model.length={self.config.model.length}, expected {SEQ_LEN}')
    if int(self.block_size) != BLOCK_SIZE:
      errors.append(f'block_size={self.block_size}, expected {BLOCK_SIZE}')
    if int(self.vocab_size) != VOCAB_SIZE:
      errors.append(f'vocab_size={self.vocab_size}, expected {VOCAB_SIZE}')
    if int(self.mask_index) != MASK_ID:
      errors.append(f'mask_id={self.mask_index}, expected {MASK_ID}')
    if int(self.tokenizer.pad_token_id) != PAD_ID:
      errors.append(
        f'pad_id={self.tokenizer.pad_token_id}, expected {PAD_ID}')
    if errors:
      raise ValueError(
        'Invalid BD3LMs-Music-V2 configuration: ' + '; '.join(errors))
    if self.config.mode == 'sample_eval':
      if self.config.model.attn_backend == 'flex':
        raise ValueError(
          'XMIDI-GetMusic sample_eval requires model.attn_backend=sdpa; '
          'FlexAttention is used only for the 2L training layout.')
      if not self.config.sampling.kv_cache:
        raise ValueError(
          'XMIDI-GetMusic sample_eval requires sampling.kv_cache=true; '
          'the source-conditioned sampler uses absolute-position cached '
          'attention and does not permit the legacy no-cache mask.')
      if not self.config.sampling.first_hitting:
        raise ValueError(
          'XMIDI-GetMusic sample_eval requires first-hitting=true for the '
          'canonical audited inference protocol.')
      if (self.config.sampling.first_hitting
          and int(self.config.algo.T) < int(self.block_size)):
        raise ValueError(
          f'first-hitting requires algo.T >= block_size ({self.block_size}); '
          'each reverse step resolves at most one token.')

  def _get_parameters(self):
    parameters = [self.backbone.parameters(),
                  self.noise.parameters()]
    return itertools.chain(* parameters)

  def on_validation_model_zero_grad(self) -> None:
    '''
    Small hack to avoid first validation on resume. 
    This will NOT work if the gradient accumulation step should be performed at this point.
    '''
    super().on_validation_model_zero_grad()
    if self.trainer.ckpt_path is not None and getattr(self, '_restarting_skip_val_flag', True):
        self.trainer.sanity_checking = True
        self._restarting_skip_val_flag = False

  def _validate_configuration(self):
    if self.config.mode == 'sample_eval' and \
        self.config.sampling.first_hitting:
      assert self.config.loader.eval_batch_size == 1
    assert self.config.algo.backbone in {
      'dit', 'ar', 'hf_dit'}
    if self.config.algo.parameterization == 'ar':
      assert not self.config.algo.time_conditioning
    if self.config.sampling.kv_cache:
      assert self.config.algo.name in {'ar', 'bd3lm'}
      
    if self.parameterization in {'sedd'}:
      assert self.time_conditioning
    
    if self.config.mode == 'sample_eval':
      assert self.config.model.attn_backend != 'flex', 'FlexAttention mask not supported at inference.'
    if self.config.model.attn_backend == 'flex':
      assert self.config.algo.name == 'bd3lm', 'Custom FlexAttention mask only supported for BD3LM.'
      
  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs) 
    self.metrics.to(*args, **kwargs)
    if hasattr(self.backbone, "block_diff_mask") and self.config.model.attn_backend == 'sdpa':
      self.backbone.block_diff_mask = self.backbone.block_diff_mask.to(*args, **kwargs)
    elif hasattr(self.backbone, "block_diff_mask") and self.config.model.attn_backend == 'flex':
      self.backbone.block_diff_mask = self.backbone.block_diff_mask.to(self.device)
    if hasattr(self, 'sampling_eps_min') and torch.is_tensor(self.sampling_eps_min):
      self.sampling_eps_min = self.sampling_eps_min.to(*args, **kwargs)
      self.sampling_eps_max = self.sampling_eps_max.to(*args, **kwargs)
    return self

  def _replace_ckpt_keys(self, checkpoint):
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k,v in state_dict.items():
      new_state_dict[k.replace('_orig_mod.', '')] = v
    checkpoint['state_dict'] = new_state_dict
    return checkpoint

  def on_load_checkpoint(self, checkpoint):
    print('Loading checkpoint at', checkpoint['global_step'])
    self._restarting_skip_val_flag = True

    # for models compiled with `torch.compile`
    if '_orig_mod.' in list(checkpoint['state_dict'].keys())[0]:
      checkpoint = self._replace_ckpt_keys(checkpoint)

    if self.ema:
      if 'ema' in checkpoint:
        self.ema.load_state_dict(checkpoint['ema'])
      else:
        # A newly initialised EMA shadow belongs to the random pre-load model,
        # not to this checkpoint. Disable it rather than silently evaluating
        # random shadow weights.
        print('[on_load_checkpoint] No EMA state in checkpoint; disabling EMA.')
        self.ema = None
    if 'sampling_eps_min' in checkpoint.keys():
      self.sampling_eps_min = checkpoint['sampling_eps_min']
      self.sampling_eps_max = checkpoint['sampling_eps_max']
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
    self.fast_forward_epochs = checkpoint['loops'][
      'fit_loop']['epoch_progress']['current']['completed']
    self.fast_forward_batches = checkpoint['loops'][
      'fit_loop']['epoch_loop.batch_progress'][
        'current']['completed']

  def on_save_checkpoint(self, checkpoint):
    if self.ema:
      checkpoint['ema'] = self.ema.state_dict()
    if hasattr(self, 'sampling_eps_min'):
      checkpoint['sampling_eps_min'] = self.sampling_eps_min
      checkpoint['sampling_eps_max'] = self.sampling_eps_max
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
    # ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
    # behind, so we're using the optimizer's progress.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['total'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total'][
              'completed'] * self.trainer.accumulate_grad_batches
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['current'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['current'][
              'completed'] * self.trainer.accumulate_grad_batches
    # _batches_that_stepped tracks the number of global steps, not the number
    # of local steps, so we don't multiply with self.trainer.accumulate_grad_batches here.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.state_dict'][
        '_batches_that_stepped'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total']['completed']
    if 'sampler' not in checkpoint.keys():
      checkpoint['sampler'] = {}
    if hasattr(self.trainer.train_dataloader.sampler,
               'state_dict'):
      sampler_state_dict = self.trainer.\
        train_dataloader.sampler.state_dict()
      checkpoint['sampler'][
        'random_state'] = sampler_state_dict.get(
          'random_state', None)
    else:
      checkpoint['sampler']['random_state'] = None

  def on_train_start(self):
    if self.ema:
      self.ema.move_shadow_params_to_device(self.device)
    # Adapted from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
    distributed = (
      self.trainer._accelerator_connector.use_distributed_sampler
      and self.trainer._accelerator_connector.is_distributed)
    if distributed:
      sampler_cls = dataloader.FaultTolerantDistributedSampler
    else:
      sampler_cls = dataloader.RandomFaultTolerantSampler
    updated_dls = []
    for dl in self.trainer.fit_loop._combined_loader.flattened:
      if hasattr(dl.sampler, 'shuffle'):
        dl_sampler = sampler_cls(
          dl.dataset, shuffle=dl.sampler.shuffle)
      else:
        dl_sampler = sampler_cls(dl.dataset)
      if (distributed
          and self.fast_forward_epochs is not None
          and self.fast_forward_batches is not None):
        dl_sampler.load_state_dict({
          'epoch': self.fast_forward_epochs,
          'counter': (self.fast_forward_batches
                      * self.config.loader.batch_size)})
      updated_dls.append(
        torch.utils.data.DataLoader(
          dl.dataset,
          batch_size=self.config.loader.batch_size,
          num_workers=self.config.loader.num_workers,
          pin_memory=self.config.loader.pin_memory,
          sampler=dl_sampler,
          shuffle=False,
          persistent_workers=self.config.loader.num_workers > 0))
    self.trainer.fit_loop._combined_loader.flattened = updated_dls

  def optimizer_step(self, *args, **kwargs):
    super().optimizer_step(*args, **kwargs)
    if self.ema:
      self.ema.update(self._get_parameters())

  def _subs_parameterization(self, logits, xt):
    # log prob at the mask index = - infinity
    logits[:, :, self.mask_index] += self.neg_infinity
    
    # Normalize the logits such that x.exp() is
    # a probability distribution over vocab_size.
    logits = logits - torch.logsumexp(logits, dim=-1,
                                      keepdim=True)
    
    # Apply updates directly in the logits matrix.
    # For the logits of the unmasked tokens, set all values
    # to -infinity except for the indices corresponding to
    # the unmasked tokens.
    unmasked_indices = (xt != self.mask_index)
    logits[unmasked_indices] = self.neg_infinity
    logits[unmasked_indices, xt[unmasked_indices]] = 0
    return logits

  def _sedd_parameterization(self, logits, xt, sigma):
    esigm1_log = torch.where(
      sigma < 0.5,
      torch.expm1(sigma),
      sigma.exp() - 1).log().to(logits.dtype)
    # logits shape
    # (batch_size, diffusion_model_input_length, vocab_size)
    logits = logits - esigm1_log[:, None, None] - np.log(
      logits.shape[-1] - 1)
    # The below scatter operation sets the log score
    # for the input word to 0.
    logits = torch.scatter(logits, -1, xt[..., None],
                           torch.zeros_like(logits[..., :1]))
    return logits

  def _process_sigma(self, sigma):
    # cause of overfitting for block size 1?
    if self.parameterization == 'ar':
      return None
    assert sigma.ndim == 2
    sigma = sigma.mean(-1).squeeze()
    if sigma.ndim == 0:
      sigma = sigma.unsqueeze(0)
    if not self.time_conditioning:
      sigma = torch.zeros_like(sigma)
    assert sigma.ndim == 1, sigma.shape
    return sigma

  def forward(self, x, sigma, sample_mode=False, store_kv=False):
    """Returns log score."""
    sigma = self._process_sigma(sigma)
    with torch.amp.autocast('cuda', dtype=torch.float32):
      if self.config.algo.name == 'bd3lm':
        logits = self.backbone(x, sigma,
                              store_kv=store_kv,
                              sample_mode=sample_mode)
      elif self.config.algo.name == 'ar':
        if self.config.algo.backbone == 'hf_dit':
          logits = self.backbone(x, None)     
        else:
          logits = self.backbone(x, sigma, sample_mode=sample_mode, store_kv=store_kv)
        logits[:, :, self.mask_index] = self.neg_infinity
        logits = logits.log_softmax(-1)
      else:
        logits = self.backbone(x, sigma)

    if self.cross_attn:
      x = x[:, :self.config.model.length]
    # `<UNK>` is a DAW-supplied structural boundary in V2, never a token to
    # synthesize. Fixed unmasked `<UNK>` cells are restored by SUBS below.
    if (sample_mode
        and getattr(getattr(self.config, 'data', None), 'train', None)
        == 'xmidi_getmusic'):
      from xmidi_getmusic.constants import UNK_ID
      logits[:, :, UNK_ID] = self.neg_infinity
    if self.parameterization == 'subs':
      return self._subs_parameterization(logits=logits,
                                      xt=x)
    elif self.parameterization == 'sedd':
      return self._sedd_parameterization(logits=logits,
                                        xt=x,
                                        sigma=sigma)
    return logits

  def _selected_subs_log_probs(self, x, sigma, targets, output_positions):
    """Exact SUBS target log-probabilities without dense sequence logits."""
    sigma = self._process_sigma(sigma)
    with torch.amp.autocast('cuda', dtype=torch.float32):
      return self.backbone(
        x,
        sigma,
        target_ids=targets,
        output_positions=output_positions,
        excluded_token_id=self.mask_index)
    
  def on_train_epoch_start(self):
    self.backbone.train()
    self.noise.train()
    self.metrics.reset()
    assert self.metrics.train_nlls.nll.mean_value == 0
    assert self.metrics.train_nlls.nll.weight == 0

  def training_step(self, batch, batch_idx):
    del batch_idx
    losses = self._loss(batch['input_ids'],
                        batch['attention_mask'],
                        row_unk_mask=batch.get('row_unk_mask'))
    self.metrics.train_nlls.update(losses.nlls, losses.token_mask)
    self.log(name='trainer/loss',
             value=losses.loss.item(),
             on_step=True,
             on_epoch=False,
             sync_dist=True)
    return losses.loss

  def on_validation_epoch_start(self):
    self.metrics.reset()
    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      self.ema.copy_to(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.eval()
    self.backbone.eval()
    self.noise.eval()
    assert self.metrics.valid_nlls.nll.mean_value == 0
    assert self.metrics.valid_nlls.nll.weight == 0
    self.sampling_eps = self.config.training.sampling_eps

  def on_validation_epoch_end(self):
    for k, v in self.metrics.valid_nlls.items():
      self.log(name=k,  value=v.compute(), on_step=False,
              on_epoch=True, sync_dist=True)
    if self.ema:
      self.ema.restore(self._get_parameters())
    if self.var_min and not self.trainer.sanity_checking:
      self._clipped_schedule_search()
      self.log('sampling_eps_min',
               self.sampling_eps_min,
               on_epoch=True,
               on_step=False,
               sync_dist=True)
      self.log('sampling_eps_max',
               self.sampling_eps_max,
               on_epoch=True,
               on_step=False,
               sync_dist=True)
  
  def _check_val_sampling_intvl(self, sampling_eps_min, sampling_eps_max):
    """Checks if the current sampling interval is valid for reporting likelihood."""
    if (sampling_eps_min == 1e-3 \
        and sampling_eps_max == 1 \
        and not (self.block_size == 1 and self.config.training.eval_nll)):
      return True # elbo
    elif (self.block_size == 1 and sampling_eps_min >= 1):
      return True # nll (block size 1)
    return False # not a valid elbo (biased estimate)
      
  def validation_step(self, batch, batch_idx):
    row_unk_mask = batch.get('row_unk_mask')
    if self.var_min:
      for noise_clip_start in self.metrics.valid_vars.keys():
        sampling_eps_min, sampling_eps_max = noise_clip_start
        if self._check_val_sampling_intvl(sampling_eps_min, sampling_eps_max) == True:
          # compute and record nelbo
          losses_clip = self._loss(batch['input_ids'],
                            batch['attention_mask'],
                            sampling_eps_min=sampling_eps_min,
                            sampling_eps_max=sampling_eps_max,
                            row_unk_mask=row_unk_mask)
          losses = Loss(
            nlls=losses_clip.nlls.clone(),
            token_mask=losses_clip.token_mask,
            loss=losses_clip.loss.clone())
        elif len(self.metrics.valid_vars[noise_clip_start]) < 100:
          # elbo from clipped schedule (biased estimate)
          losses_clip = self._loss(batch['input_ids'],
                            batch['attention_mask'],
                            sampling_eps_min=sampling_eps_min,
                            sampling_eps_max=sampling_eps_max,
                            row_unk_mask=row_unk_mask)
        if len(self.metrics.valid_vars[noise_clip_start]) < 100:
          # only report variance over 100 batches
          nlls = losses_clip.nlls
          self.metrics.valid_vars[noise_clip_start].append(
            nlls.reshape(
              nlls.shape[0], -1, self.block_size).mean(-1))
    elif self.block_size == 1:
      # nll
      losses = self._loss(batch['input_ids'],
                          batch['attention_mask'],
                          sampling_eps_min=1,
                          sampling_eps_max=1,
                          row_unk_mask=row_unk_mask)
    else:
      # nelbo
      losses = self._loss(batch['input_ids'],
                          batch['attention_mask'],
                          sampling_eps_min=1e-3,
                          sampling_eps_max=1,
                          row_unk_mask=row_unk_mask)
    self.metrics.valid_nlls.update(losses.nlls, losses.token_mask)
    return losses.loss

  def configure_optimizers(self):
    # TODO(yair): Lightning currently giving this warning when using `fp16`:
    #  "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
    #  Not clear if this is a problem or not.
    #  See: https://github.com/Lightning-AI/pytorch-lightning/issues/5558
    optimizer = torch.optim.AdamW(
      self._get_parameters(),
      lr=self.config.optim.lr,
      betas=(self.config.optim.beta1,
             self.config.optim.beta2),
      eps=self.config.optim.eps,
      weight_decay=self.config.optim.weight_decay)

    scheduler = hydra.utils.instantiate(
      self.config.lr_scheduler, optimizer=optimizer)
    scheduler_dict = {'scheduler': scheduler,
                      'interval': 'step',
                      'monitor': 'val/loss',
                      'name': 'trainer/lr'}
    return [optimizer], [scheduler_dict]
  
  def _resample_q_xt(
      self, x, xt, move_indices, p, block_size, sampling_eps_min, sampling_eps_max):
    """Resamples x_t if the percentage of masked tokens is outside the bounds
    defined by sampling_eps_min and sampling_eps_max."""
    perc_masked = (xt == self.mask_index).float().sum(-1) / block_size
    while (perc_masked < sampling_eps_min).any() or \
      (perc_masked > sampling_eps_max).any():
      # if a bound is epsilon, don't resample
      if sampling_eps_min == 1e-3 and sampling_eps_max != 1:
        regen_idx = (perc_masked > sampling_eps_max)
        if regen_idx.max() == 0:
          break
      elif sampling_eps_min != 1e-3 and sampling_eps_max == 1:
        regen_idx = (perc_masked < sampling_eps_min)
        if regen_idx.max() == 0:
          break
      elif sampling_eps_min != 1e-3 and sampling_eps_max != 1:
        regen_idx = (perc_masked < sampling_eps_min) | (perc_masked > sampling_eps_max)
      regen_idx = regen_idx.repeat_interleave(block_size,dim=-1)
      move_indices[regen_idx] = (torch.rand(
        * x.shape, device=x.device) < p)[regen_idx]
      xt = torch.where(move_indices, self.mask_index, x)
      xt = xt.reshape(xt.shape[0], -1, block_size)
      perc_masked = (xt == self.mask_index).float().sum(-1) / block_size
    return xt
  
  def q_xt(
      self, x, p, block_size=None, sampling_eps_min=None, sampling_eps_max=None,
      region_mask=None):
    """Computes the noisy sample xt.

    Args:
      x: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input. 
      p: float torch.Tensor with shape (batch_size, 1).
      block_size: int, block size.
      sampling_eps_min: float, minimum percentage of masked tokens.
      sampling_eps_max: float, maximum percentage of masked tokens.
      region_mask: optional float/bool torch.Tensor with shape (batch_size,
          diffusion_model_input_length). When given, masking is restricted to
          positions where region_mask == 1; positions with region_mask == 0
          stay clean (== x) throughout the forward diffusion process. This is
          how the XMIDI dataset enforces "source rows / prompt cols / <UNK>
          positions are never masked" (see R&D §"自适应掩蔽算法").
          Default None preserves the original BD3-LM behaviour (unrestricted
          masking).
    """
    if block_size is None:
      block_size = self.block_size
  
    move_indices = torch.rand(
      * x.shape, device=x.device) <= p
    if region_mask is not None:
      # XMIDI: forbid masking outside the loss region.
      move_indices = move_indices & region_mask.bool()

    xt = torch.where(move_indices, self.mask_index, x)

    if block_size == 1 and sampling_eps_min == 1.0:
      # full-mask shortcut. Honour region_mask if provided.
      if region_mask is not None:
        return torch.where(region_mask.bool(),
                            torch.full_like(x, self.mask_index), x)
      return torch.full_like(x, self.mask_index)

    # no need to resample for bounds 1e-3, 1
    if self.config.training.resample and \
      not (sampling_eps_min == 1e-3 and sampling_eps_max == 1.0):
      xt = xt.reshape(xt.shape[0], -1, block_size)
      xt = self._resample_q_xt(x,
                               xt,
                               move_indices,
                               p,
                               block_size,
                               sampling_eps_min,
                               sampling_eps_max)
      xt = xt.reshape(xt.shape[0], -1)

    # Out-of-region guard: even after _resample_q_xt may have stomped on
    # extra positions, force every region_mask==0 position back to x.
    if region_mask is not None:
      xt = torch.where(region_mask.bool(), xt, x)

    return xt

  def _sample_prior(self, *batch_dims):
    return self.mask_index * torch.ones(
      * batch_dims, dtype=torch.int64, device=self.device)

  @torch.no_grad()
  def _nucleus_sample(self, p_x0):
    p = float(self.config.sampling.nucleus_p)
    if not 0.0 < p <= 1.0:
      raise ValueError(f'nucleus_p must be in (0, 1], got {p}')
    if p == 1.0:
      return p_x0
    p_x0_ = p_x0[:, -self.block_size:].clone()
    sorted_probs, sorted_indices = p_x0_.sort(dim=-1, descending=True)
    cumulative = sorted_probs.cumsum(dim=-1)
    # Standard nucleus sampling keeps the first token that crosses p. Shift
    # the removal mask right so the retained set has cumulative mass >= p.
    remove = cumulative > p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_probs = sorted_probs.masked_fill(remove, 0.0)
    p_x0_.zero_().scatter_(-1, sorted_indices, sorted_probs)
    p_x0_ /= p_x0_.sum(-1, keepdim=True).clamp_min(1e-30)
    p_x0[:, -self.block_size:] = p_x0_
    return p_x0

  @torch.no_grad()
  def _xmidi_commit_kv_block(self, tokens, sigma):
    """Commit one clean V2 block without allocating 142421-way logits."""
    processed_sigma = self._process_sigma(sigma)
    no_projection = torch.zeros_like(tokens, dtype=torch.bool)
    with torch.amp.autocast('cuda', dtype=torch.float32):
      self.backbone(
        tokens,
        processed_sigma,
        sample_mode=True,
        store_kv=True,
        target_ids=tokens,
        output_positions=no_projection,
        excluded_token_id=self.mask_index)

  @torch.no_grad()
  def _ddpm_caching_update(self, x, t, dt, p_x0=None):
    _, move_chance_t = self.noise(t)
    _, move_chance_s = self.noise((t - dt).clamp_min(0.0))
    sigma_t = self._sigma_from_p(move_chance_t)
    move_chance_t = move_chance_t[:, None]
    move_chance_s = move_chance_s[:, None]
    mask_prob = move_chance_s / move_chance_t

    if p_x0 is None:
      if self.config.sampling.kv_cache:
        p_x0 = self.forward(x[:, -self.block_size:],
                        sigma_t,
                        sample_mode=True).to(torch.float64)
      else:   
        p_x0 = self.forward(x,
                          sigma_t,
                          sample_mode=True).to(torch.float64)
        p_x0 = p_x0[:, -self.block_size:]
      p_x0 = p_x0.exp()
      p_x0 = self._nucleus_sample(p_x0)

    if self.config.sampling.first_hitting:
      x_block = _sample_categorical(p_x0)
      # Select one unresolved position independently for every sample. This
      # supports mixed batches in which some samples have already completed.
      current_block = x[:, -self.block_size:]
      unresolved = current_block == self.mask_index
      selected = torch.zeros_like(unresolved)
      for batch_idx in range(unresolved.shape[0]):
        candidates = unresolved[batch_idx].nonzero(as_tuple=False).flatten()
        if candidates.numel() > 0:
          choice = torch.randint(
            candidates.numel(), (1,), device=x.device)
          selected[batch_idx, candidates[choice]] = True
      x_block = torch.where(selected, x_block, current_block)
    else:
      q_xs = p_x0 * (1 - mask_prob)
      q_xs[:, :, self.mask_index] = mask_prob.squeeze(-1)
      x_block = _sample_categorical(q_xs)
    copy_flag = (x[:, -self.block_size:] != self.mask_index).to(x.dtype)
    x_block =  copy_flag * x[:, -self.block_size:] + (1 - copy_flag) * x_block
    x_new = torch.cat((x[:, :-self.block_size], x_block), dim=-1)

    # Compute KV cache if all tokens in a block are sampled. V2 uses a
    # no-projection commit path because logits for a clean block are discarded.
    if self.config.sampling.kv_cache and self.mask_index not in x_block:
      data_name = getattr(getattr(self.config, 'data', None), 'train', None)
      if data_name == 'xmidi_getmusic':
        self._xmidi_commit_kv_block(x_block, sigma_t)
      else:
        _ = self.forward(x_block, sigma_t, sample_mode=True, store_kv=True)

    if not torch.allclose(x_new, x):
      return None, x_new
    else:
      return p_x0, x_new

  @torch.no_grad()
  def _ar_sampler(self, bsz, context_len=1024):
    # reset kvs
    if self.config.sampling.kv_cache:
      self.backbone.reset_kv_cache()

    with torch.amp.autocast('cuda', dtype=torch.float32):
      # precompute token buffer
      num_pred_tokens = self.num_tokens - 1
      x = torch.zeros(
        (bsz, num_pred_tokens + 1),
        dtype=torch.long,
        device=self.device)
      x[:, 0] = self.tokenizer.bos_token_id
      stop = False
      for i in tqdm(range(num_pred_tokens)):
        # need to sample a gumbel for each token
        # to save memory in variable-length sampling
        noise = (torch.distributions.Gumbel(0, 1)
                .sample((bsz, self.vocab_size))
                .to(self.device))
        next_logits = self.forward(
          x[:, :i + 1][:, -context_len:],
          None,
          store_kv=self.config.sampling.kv_cache)[:, -1:].to(torch.float64)
    
        next_logits = next_logits.exp()
        next_logits = self._nucleus_sample(next_logits).log()
        y = (next_logits[:, -1] + noise).argmax(-1)
        # check if we need to resample (or stop sampling for variable-length sampling)
        if (i+1) > 256:
          stop, x_out = self._check_stop_conds(x[:, :i+1])
          if stop:
            x = x_out
        if (stop and not self.config.sampling.var_length) \
          or (stop and x.shape[-1] == 1):
          return None
        elif stop:
          break
        x[:, i + 1] = y
      return x
  
  @torch.no_grad()
  def _sample(
    self, seqlen=None, num_steps=None, eps=1e-5, batch_size_per_gpu=None):
    """Generate samples from the model."""
    if seqlen is None:
      seqlen = self.config.model.length
    if batch_size_per_gpu is None:
      batch_size_per_gpu = self.config.loader.eval_batch_size

    # XMIDI: source-conditioned semi-AR generation. Pulls source/prompts
    # from the validation loader and skips _check_stop_conds (G3 + G4).
    # Same code path is used for the new ``xmidi_getmusic`` variant: only
    # constants (NUM_ROWS, NUM_COLS, UNK_ID) differ, all dispatched via
    # ``_xmidi_module()``.
    from xmidi_compat import is_xmidi_like
    if is_xmidi_like(self.config):
      decoded, _ = self._xmidi_sample_full(
        num_steps=num_steps,
        batch_size_per_gpu=batch_size_per_gpu)
      return decoded

    samples = []
    if self.parameterization == 'ar':
      for _ in range(self.config.sampling.num_sample_batches):
        sample_i, num_tries = None, 0
        while sample_i is None:
          num_tries += 1
          sample_i = self._ar_sampler(batch_size_per_gpu)
          if num_tries > 10:
            raise ValueError('Sampling failed.')
        samples.append(sample_i)
        self.metrics.gen_nfes.append(self.config.model.length)
      samples = torch.cat(samples, dim=0) 
      return self.tokenizer.batch_decode(samples)
    if self.sampler == 'semi_ar':
      for _ in range(self.config.sampling.num_sample_batches):
        sample_i, num_tries = None, 0
        while sample_i is None:
          num_tries += 1
          sample_i, nfes = self._semi_ar_sampler(
            n_samples=batch_size_per_gpu,
            num_strides=(seqlen // self.block_size), 
            num_steps=num_steps,
            seqlen=seqlen)
          if num_tries > 10:
            raise ValueError('Sampling failed.')
        samples.append(sample_i)
        self.metrics.nfes.update(nfes)
        self.metrics.gen_nfes.append(nfes)
    else:
      nfes = num_steps
      for _ in range(self.config.sampling.num_sample_batches):
        sample_i, num_tries = None, 0
        while sample_i is None:
          sample_i = self._analytic_sampler(
            n_samples=batch_size_per_gpu,
            num_steps=num_steps,
            seqlen=seqlen,
            eps=eps)
          num_tries += 1
          if num_tries > 10 and sample_i is None:
            raise ValueError('Sampling failed.')
        samples.append(sample_i)
        self.metrics.nfes.update(nfes)
        self.metrics.gen_nfes.append(nfes)
    if self.config.sampling.var_length:
      # variable-length samples are truncated at EOS / low-entropy
      # independently per batch, so their lengths can differ.
      # Decode each batch separately, then flatten the list.
      decoded = []
      for s in samples:
        decoded.extend(self.tokenizer.batch_decode(s))
      return decoded
    samples = torch.cat(samples, dim=0)
    return self.tokenizer.batch_decode(samples)

  def _sigma_from_p(self, p):
    return torch.min(- torch.log(1 - p), self.noise.sigma_max)

  def restore_model_and_sample(self, num_steps, eps=1e-5, seqlen=None):
    """Generate samples from the model."""
    if self.ema:  
      self.ema.store(self._get_parameters())
      self.ema.copy_to(self._get_parameters())
    self.backbone.eval()
    self.noise.eval()
    samples = self._sample(
      seqlen=seqlen,
      batch_size_per_gpu=self.config.loader.eval_batch_size,
      num_steps=num_steps,
      eps=eps)
    # XMIDI / XMIDI-GetMusic are symbolic music; GPT-2 generative
    # perplexity does not apply.
    from xmidi_compat import is_xmidi_like
    if not is_xmidi_like(self.config):
      self.metrics.record_generative_perplexity(
        samples,
        self.config.model.length,
        self.config.loader.eval_batch_size,
        self.device)
    return samples

  def get_score(self, x, sigma):
    model_output = self.forward(x, sigma).to(torch.float64)
    if self.config.sampling.nucleus_p == 1.0:
      return model_output.exp()
    model_output = model_output - model_output.logsumexp(-1, keepdim=True)
    model_output = self._nucleus_sample(model_output.exp())
    return model_output

  def _staggered_score(self, score, dsigma):
    score = score.clone()
    extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
    score *= dsigma.exp()[:, None]
    score[..., self.mask_index] += extra_const
    return score

  def _analytic_update(self, x, t, dt):
    sigma_t = self._sigma_from_p(self.noise(t)[1])
    sigma_s = self._sigma_from_p(self.noise(t - dt)[1])
    dsigma = sigma_t - sigma_s
    score = self.get_score(x, sigma_t)
    stag_score = self._staggered_score(score, dsigma)
    probs = stag_score * self._transp_transition(x, dsigma)
    return _sample_categorical(probs)


  def _denoiser_update(self, x, t):
    sigma = self._sigma_from_p(self.noise(t)[1])
    score = self.get_score(x, sigma)
    stag_score = self._staggered_score(score, sigma)
    probs = stag_score * self._transp_transition(x, sigma)
    probs[..., self.mask_index] = 0
    samples = _sample_categorical(probs)
    return samples


  def _transp_transition(self, i, sigma):
    sigma = _unsqueeze(sigma, reference=i[..., None])
    edge = torch.exp(-sigma) * F.one_hot(
      i, num_classes=self.vocab_size)
    edge += torch.where(i == self.mask_index,
                        1 - torch.exp(-sigma).squeeze(-1),
                        0)[..., None]
    return edge

  def _sample_t(
      self, batch_dims, device, sampling_eps_min, sampling_eps_max, block_size=None):
    if block_size is None:
      block_size = self.block_size
    n = batch_dims[-1]
    num_blocks = n // block_size
    _eps_b = torch.rand((batch_dims[0], num_blocks), device=device)

    # antithetic sampling along blocks & batches (for uniform sampling)
    if self.antithetic_sampling:
      offset_b = torch.arange(batch_dims[0] * num_blocks, device=device) / (batch_dims[0] * num_blocks)
      offset_b = offset_b.view(batch_dims[0], num_blocks)
      _eps_b = (_eps_b / (batch_dims[0] * num_blocks) + offset_b) % 1
    t = _eps_b
    if block_size != self.config.model.length:
      t = t.repeat_interleave(block_size, dim=-1)

    # nll
    if sampling_eps_max >= 1 and sampling_eps_min >= 1:
      return torch.ones_like(t)
    t = t * (sampling_eps_max - sampling_eps_min) + sampling_eps_min
    return t

  def _maybe_sub_sample(self, x0, attention_mask):
    seqlen = x0.shape[1]
    if seqlen > self.num_tokens:
      assert seqlen == 2 * self.num_tokens
      # cropping is needed for text8-crop dataset
      # try the same starting point for now
      start = np.random.choice(self.num_tokens)
      end = start + self.num_tokens
      input_tokens = x0[:, start: end]
      output_tokens = x0[:, start + 1: end + 1]
      new_attention_mask = attention_mask[:, start: end]

      # Helps with validation ppl, since the val
      # examples will all start and end with BOS/EOS
      if self.config.data.insert_train_special == True:
        input_tokens[:, 0] = self.tokenizer.bos_token_id
        output_tokens[:, -1] = self.tokenizer.eos_token_id
    elif self.parameterization == 'ar':
      input_tokens = x0[:, :-1]
      output_tokens = x0[:, 1:]
      new_attention_mask = attention_mask[:, 1:]
    else:
      input_tokens = x0
      output_tokens = None
      new_attention_mask = attention_mask
    
    return input_tokens, output_tokens, new_attention_mask

  def _forward_pass_diffusion(self, x0, t=None, sampling_eps_min=None,
                              sampling_eps_max=None, region_mask=None):
    if t is None:
      t = self._sample_t(x0.shape,
                         x0.device,
                         sampling_eps_min,
                         sampling_eps_max)

    loss_scale, p = self.noise(t)
    sigma = self._sigma_from_p(p[:,0].unsqueeze(-1))
    dsigma = - loss_scale * torch.expm1(sigma) # used for sedd

    # below is needed to reproduce mdlm/sedd numbers with models from sahoo et al
    # (numerical imprecision computing probs under loglinear schedule)
    if self.mdlm_loss_scale:
      sigma, dsigma = self.noise.total_noise(t), self.noise.rate_noise(t)
      p = 1 - torch.exp(-sigma)
      loss_scale = - (dsigma / torch.expm1(sigma))

    xt = self.q_xt(x0,
                   p,
                   sampling_eps_min=sampling_eps_min,
                   sampling_eps_max=sampling_eps_max,
                   region_mask=region_mask)
    if sampling_eps_min is not None and sampling_eps_min > 0.5:
      loss_scale = - torch.ones_like(loss_scale)
    if self.ignore_bos:
      xt[:, 0] = x0[:, 0]
    
    x_input = xt
    if self.cross_attn:
      x_input = torch.cat((xt, x0), dim=-1)

    # V2's 142,421-way classifier is evaluated only where a generative token
    # is currently masked. Unmasked SUBS positions have log p(x0)=0 exactly.
    from xmidi_compat import is_xmidi_like
    if (is_xmidi_like(self.config)
        and self.parameterization == 'subs'
        and region_mask is not None):
      output_positions = (xt == self.mask_index) & region_mask.bool()
      log_p_theta = self._selected_subs_log_probs(
        x_input, sigma, x0, output_positions)
      utils.print_nans(log_p_theta, 'selected_log_p_theta')
      return loss_scale * log_p_theta

    model_output = self.forward(x_input, sigma=sigma)
    utils.print_nans(model_output, 'model_output')

    if self.parameterization == 'sedd':
      return dsigma * self._score_entropy(
        model_output, sigma, xt, x0)

    log_p_theta = torch.gather(
      input=model_output,
      dim=-1,
      index=x0[:, :, None]).squeeze(-1)
    loss = loss_scale * log_p_theta
    return loss

  def _loss(self, x0, attention_mask, t=None, sampling_eps_min=None,
            sampling_eps_max=None, row_unk_mask=None):
    if sampling_eps_min is None and hasattr(self, 'sampling_eps_min'):
      sampling_eps_min = self.sampling_eps_min
      sampling_eps_max = self.sampling_eps_max
    elif not hasattr(self, 'sampling_eps_min'):
      sampling_eps_min = 1e-3
      sampling_eps_max = 1.0
    (input_tokens, output_tokens,
     attention_mask) = self._maybe_sub_sample(
       x0, attention_mask)

    # XMIDI: register the per-batch row-UNK mask on the backbone so that
    # ``DIT.forward`` can swap in a per-sample attention mask that drops
    # entirely-<UNK> rows. Cleared in the finally block to keep state-free
    # contracts elsewhere (sampling, validation across loops, etc.).
    set_row_unk = (row_unk_mask is not None
                   and hasattr(self.backbone, 'set_xmidi_row_unk_mask'))
    if set_row_unk:
      self.backbone.set_xmidi_row_unk_mask(row_unk_mask)
    try:
      if self.parameterization == 'ar':
        output = self.forward(input_tokens, None)
        loss = - output.gather(
          -1, output_tokens[:, :, None])[:, :, 0]
      else:
        # XMIDI: ``attention_mask`` already encodes the loss/diffusion region
        # (1 = "may be masked + counted in NLL", 0 = "fixed context, e.g.
        # source rows / prompt cols / <UNK>"). Plumb it into q_xt so that
        # forbidden positions stay clean in xt, mirroring R&D §"自适应掩蔽".
        # For non-XMIDI datasets attention_mask is the standard padding mask
        # and we leave the original (unrestricted) masking behaviour intact.
        from xmidi_compat import is_xmidi_like
        use_region = is_xmidi_like(self.config)
        region_mask = attention_mask if use_region else None
        loss = self._forward_pass_diffusion(
          input_tokens,
          sampling_eps_min=sampling_eps_min,
          sampling_eps_max=sampling_eps_max,
          region_mask=region_mask,)
    finally:
      if set_row_unk:
        self.backbone.set_xmidi_row_unk_mask(None)
    
    if self.ignore_bos and not self.training:
      attention_mask[:, 0] = 0
      
    token_mask = attention_mask.bool()
    nlls = loss * token_mask.to(loss.dtype)
    region_sizes = token_mask.sum(dim=-1)
    valid_samples = region_sizes > 0
    if not valid_samples.any():
      raise ValueError('Batch contains no valid diffusion target positions')
    per_sample_nll = nlls.sum(dim=-1) / region_sizes.clamp_min(1)
    token_nll = per_sample_nll[valid_samples].mean()
    return Loss(loss=token_nll,
                nlls=nlls,
                token_mask=token_mask)

  def _clipped_schedule_search(self):
    # collect losses per batch across devices and sum them per interval
    best_var = float('inf')
    for (eps_min, eps_max), var in self.metrics.valid_vars.items():
      all_vars = torch.tensor(0., device=self.device)
      for i in range(len(var)):
        agg_var = var[i].to(self.device)
        agg_var = self.all_gather(agg_var)
        all_vars += agg_var.var()
      if all_vars < best_var:
        best_var = all_vars
        sampling_eps_min_best = eps_min
        sampling_eps_max_best = eps_max
      self.log(f'valid_var_{round(eps_min, 2)} - {round(eps_max, 2)}',
                all_vars / len(var),
                on_epoch=True,
                on_step=False,
                sync_dist=True)
    if self.config.algo.fix_clipping == False:
      self.sampling_eps_min.fill_(sampling_eps_min_best)
      self.sampling_eps_max.fill_(sampling_eps_max_best)

  def _score_entropy(self, log_score, sigma, xt, x0):
    """Computes the SEDD loss.

    Args:
      log_score: float torch.Tensor with shape (batch_size,
          diffusion_model_input_length, vocab_size),
          log score, output of the denoising network.
      xt: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      x0: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      sigma: float torch.Tensor with shape (batch_size, 1).

    Returns:
      loss with shape (batch_size, diffusion_model_input_length)
    """
    masked_indices = xt == self.mask_index

    expsig_minus_1 = torch.expm1(sigma).expand_as(xt)
    q_ratio = 1 / expsig_minus_1[masked_indices]

    words_that_were_masked = x0[masked_indices]

    neg_term = q_ratio * torch.gather(
      log_score[masked_indices],
      -1,
      words_that_were_masked[..., None]).squeeze(-1)
    score = log_score[masked_indices].exp()
    if self.mask_index == self.vocab_size - 1:
      pos_term = score[:, :-1].sum(dim=-1)
    else:
      pos_term = score[:, : self.mask_index].sum(
        dim=-1) + score[:, self.mask_index + 1:].sum(dim=-1)
    const = q_ratio * (q_ratio.log() - 1)

    entropy = torch.zeros(* xt.shape, device=xt.device)
    entropy[masked_indices] += pos_term - neg_term + const
    return entropy

  @torch.no_grad
  def _analytic_sampler(
    self, n_samples, num_steps, seqlen, eps=1e-5): 
    x = self._sample_prior(
      n_samples,
      seqlen).to(self.device)
    x[:, 0] = self.tokenizer.bos_token_id
    timesteps = torch.linspace(
      1, eps, num_steps + 1, device=self.device)
    dt = (1 - eps) / num_steps
    for i in tqdm(range(num_steps), desc='step'):
      t = timesteps[i] * torch.ones(
        x.shape[0], 1, device=self.device)
      x = self._analytic_update(x=x, t=t, dt=dt)
    # denoising step 
    t = timesteps[-1] * torch.ones(x.shape[0], 1,
                                  device=self.device)
    x = self._denoiser_update(x=x, t=t)
    
    stop, x = self._check_stop_conds(x)
    if stop:
      return None
    return x

  @torch.no_grad
  def _semi_ar_sampler(
    self, n_samples, num_steps, num_strides, seqlen, context_size=1024):
    if seqlen is None:
      seqlen = self.config.model.length
    sampling_steps = 0
          
    mdlm_semi_ar = self.config.algo.name == 'mdlm' and self.config.model.length > self.block_size
    if mdlm_semi_ar:
      # sliding window of length 512 for mdlm semi-ar decoding
      num_strides = self.config.model.length // 512
      num_strides -= 1

    ones = torch.ones((n_samples,1), dtype=self.dtype,
                      device=self.device)
    
    # reset kvs
    if self.config.sampling.kv_cache:
      self.backbone.reset_kv_cache(eval_batch_size=self.config.loader.eval_batch_size)

    for stride_num in tqdm(range(num_strides)):
      # sample next block
      if stride_num == 0:
        x_accum = self._sample_prior(n_samples, self.block_size).to(self.device)
        x_accum[:, 0] = self.tokenizer.bos_token_id
      else:
        if mdlm_semi_ar:
          x = self._sample_prior(n_samples, 512).to(self.device)
        else:
          x = self._sample_prior(n_samples, self.block_size).to(self.device)
        x_accum = torch.cat((x_accum, x), dim=1)

      # compute logits in a sliding window (context passed to model can't exceed context_size)
      end_idx = (stride_num + 1) * self.block_size
      start_idx = max(end_idx - context_size, 0)
      fwd_idx = torch.arange(start_idx, end_idx)
      if mdlm_semi_ar and stride_num > 0: # MDLM
        fwd_idx = torch.arange(512*(stride_num), (512*(stride_num))+self.block_size)

      dt = 1 / num_steps
      p_x0_cache = None
      timesteps = torch.linspace(1, 0, num_steps, device=self.device)
      t = 1
      for i in range(num_steps):
        if self.mask_index not in x_accum:
          break

        # faster (equivalent) sampler from zheng et al (2025)
        if self.config.sampling.first_hitting:
          u = np.random.rand()
          num_masked = (x_accum[:, fwd_idx] == self.mask_index).sum(-1).item()
          t *= u**(1 / num_masked)
              
        elif not self.config.sampling.first_hitting:
          t = timesteps[i]

        p_x0_cache, x_next = self._ddpm_caching_update(
            x=x_accum[:, fwd_idx],
            t=t * ones,
            dt=dt,
            p_x0=p_x0_cache,)
        if p_x0_cache is None:
          sampling_steps += 1
       
        x_accum[:, fwd_idx] = x_next

      # check if we need to resample (or stop sampling for variable-length sampling)
      if x_accum.shape[1] > 256:
        stop, x_accum = self._check_stop_conds(x_accum)
        if (stop and not self.config.sampling.var_length) \
          or (stop and x.shape[-1] == 1):
          return None, None
        elif stop:
          break
    return x_accum, sampling_steps

  # ---------------------------------------------------------------------
  # XMIDI source-conditioned sampling (R&D §推理, milestone 6)
  #
  # V2 layout (row-major flatten, 8192 tokens): rows 0..3 are fixed source;
  # rows 4..15 contain a fixed 16-token prompt followed by target content.
  # Every structural <UNK> remains fixed. All 512 blocks are visited in order:
  # fixed blocks are committed once, while requested content blocks run the
  # first-hitting reverse process and then commit their clean K/V exactly once.
  # ---------------------------------------------------------------------
  @torch.no_grad()
  def _xmidi_sample_prior(self, source, target_prompts, row_unk_mask,
                          content_unk_mask=None, condition_grid=None,
                          generation_mask=None):
    """Construct the initial x[B,SEQ_LEN] for the XMIDI sampler.

    Args:
      source:         long tensor [B, NUM_SOURCE_ROWS, NUM_COLS] - real source
      target_prompts: long tensor [B, NUM_TARGET_ROWS, NUM_PROMPT_COLS]
      row_unk_mask:   bool tensor [B, NUM_ROWS] or None - True = entire row UNK
      content_unk_mask: bool tensor [B, NUM_TARGET_ROWS, NUM_COLS-NUM_PROMPT_COLS]
                      or None. True at fixed structural <UNK> positions.
      condition_grid: optional long tensor [B, NUM_ROWS, NUM_COLS] containing
                      every DAW-fixed cell. Values under generation_mask are
                      ignored and replaced by <MASK>.
      generation_mask: optional bool tensor [B, NUM_ROWS, NUM_COLS]. Together
                      with condition_grid this is the product-level V2 C+G API;
                      it can preserve user-authored ordinary target tokens.

    Returns:
      x: long tensor [B, SEQ_LEN], where source/prompts/<UNK> rows AND
         per-token <UNK> positions in the target-content region are
         filled with <UNK>; the remaining target-content positions hold
         self.mask_index.
    """
    from xmidi_compat import get_xmidi_module
    _xm = get_xmidi_module(self.config)
    UNK_ID = _xm.UNK_ID
    NUM_ROWS = _xm.NUM_ROWS
    NUM_COLS = _xm.NUM_COLS
    NUM_SOURCE_ROWS = _xm.NUM_SOURCE_ROWS
    NUM_PROMPT_COLS = _xm.NUM_PROMPT_COLS
    SEQ_LEN = _xm.SEQ_LEN
    if source.shape[1:] != (NUM_SOURCE_ROWS, NUM_COLS):
      raise ValueError(
        f"source must be (B, {NUM_SOURCE_ROWS}, {NUM_COLS}); got {tuple(source.shape)}")
    if target_prompts.shape[1:] != (NUM_ROWS - NUM_SOURCE_ROWS, NUM_PROMPT_COLS):
      raise ValueError(
        f"target_prompts must be (B, {NUM_ROWS - NUM_SOURCE_ROWS}, "
        f"{NUM_PROMPT_COLS}); got {tuple(target_prompts.shape)}")

    B = source.shape[0]
    device = source.device
    if (condition_grid is None) != (generation_mask is None):
      raise ValueError(
        'condition_grid and generation_mask must be provided together')

    if condition_grid is not None:
      from xmidi_getmusic.eval_protocol import validate_generation_mask
      condition_grid = condition_grid.to(device=device, dtype=torch.long)
      if tuple(condition_grid.shape) != (B, NUM_ROWS, NUM_COLS):
        raise ValueError(
          f'condition_grid must be {(B, NUM_ROWS, NUM_COLS)}, got '
          f'{tuple(condition_grid.shape)}')
      generation_mask = validate_generation_mask(
        generation_mask, condition_ids=condition_grid).to(device)
      if torch.any((condition_grid < 0) | (condition_grid > UNK_ID)):
        raise ValueError('condition_grid contains an invalid or <MASK> token')
      if not torch.equal(condition_grid[:, :NUM_SOURCE_ROWS], source):
        raise ValueError('condition_grid source differs from source argument')
      if not torch.equal(
          condition_grid[:, NUM_SOURCE_ROWS:, :NUM_PROMPT_COLS],
          target_prompts):
        raise ValueError(
          'condition_grid target prompts differ from target_prompts argument')
      grid = condition_grid.clone()
      grid[generation_mask] = self.mask_index
    else:
      grid = torch.full((B, NUM_ROWS, NUM_COLS), self.mask_index,
                        dtype=torch.long, device=device)
      grid[:, :NUM_SOURCE_ROWS] = source
      grid[:, NUM_SOURCE_ROWS:, :NUM_PROMPT_COLS] = target_prompts

    if row_unk_mask is not None:
      row_unk_mask = row_unk_mask.bool().to(device)
      if tuple(row_unk_mask.shape) != (B, NUM_ROWS):
        raise ValueError(
          f'row_unk_mask must be {(B, NUM_ROWS)}, got '
          f'{tuple(row_unk_mask.shape)}')
      if condition_grid is not None:
        observed = torch.all(condition_grid == UNK_ID, dim=-1)
        if not torch.equal(observed, row_unk_mask):
          raise ValueError(
            'row_unk_mask differs from all-<UNK> rows in condition_grid')
      unk_grid = torch.full_like(grid, UNK_ID)
      row_mask = row_unk_mask.view(B, NUM_ROWS, 1).expand_as(grid)
      grid = torch.where(row_mask, unk_grid, grid)

    if content_unk_mask is not None:
      expected_shape = (B, NUM_ROWS - NUM_SOURCE_ROWS, NUM_COLS - NUM_PROMPT_COLS)
      if tuple(content_unk_mask.shape) != expected_shape:
        raise ValueError(
          f"content_unk_mask must be {expected_shape}; "
          f"got {tuple(content_unk_mask.shape)}")
      content_mask = content_unk_mask.bool().to(device)
      content_view = grid[:, NUM_SOURCE_ROWS:, NUM_PROMPT_COLS:]
      if generation_mask is not None:
        observed_content_unk = (
          condition_grid[:, NUM_SOURCE_ROWS:, NUM_PROMPT_COLS:] == UNK_ID)
        if not torch.equal(content_mask, observed_content_unk):
          raise ValueError(
            'content_unk_mask differs from <UNK> cells in condition_grid')
        generated_content = generation_mask[:, NUM_SOURCE_ROWS:, NUM_PROMPT_COLS:]
        if torch.any(content_mask & generated_content):
          raise ValueError('generation_mask selects content_unk_mask cells')
      content_view[content_mask] = UNK_ID

    return grid.view(B, SEQ_LEN)

  @torch.no_grad()
  def _xmidi_semi_ar_sampler(self, source, target_prompts, row_unk_mask,
                             num_steps, content_unk_mask=None,
                             condition_grid=None, generation_mask=None):
    """Source-conditioned semi-AR sampler for the XMIDI grid.

    Args:
      source:         [B, NUM_SOURCE_ROWS, NUM_COLS] long
      target_prompts: [B, NUM_TARGET_ROWS, NUM_PROMPT_COLS]  long
      row_unk_mask:   [B, NUM_ROWS]    bool (or None) - fully-<UNK> rows
      num_steps:      diffusion steps per content block
      content_unk_mask: [B, NUM_TARGET_ROWS, NUM_COLS-NUM_PROMPT_COLS] bool
                      (or None). True at fixed structural <UNK> cells.
      condition_grid: optional [B,NUM_ROWS,NUM_COLS] DAW-fixed grid.
      generation_mask: optional matching bool grid; exactly these cells are
                      masked and reconstructed. Values under this mask are
                      ignored, preventing reference-token leakage.

    Returns:
      x_accum:        [B, 8192] long, fully decoded V2 sequence
      sampling_steps: int, total diffusion forward passes (NFEs)
    """
    from xmidi_compat import get_xmidi_module
    _xm = get_xmidi_module(self.config)
    UNK_ID = _xm.UNK_ID
    NUM_ROWS = _xm.NUM_ROWS
    NUM_COLS = _xm.NUM_COLS
    NUM_SOURCE_ROWS = _xm.NUM_SOURCE_ROWS
    NUM_PROMPT_COLS = _xm.NUM_PROMPT_COLS
    if num_steps is None or num_steps <= 0:
      # algo.T may default to 0; fall back to one denoising step per token.
      num_steps = max(1, self.block_size)

    block_size = self.block_size
    if NUM_PROMPT_COLS % block_size != 0:
      raise ValueError(
        f"NUM_PROMPT_COLS={NUM_PROMPT_COLS} must be a multiple of "
        f"block_size={block_size} so prompt blocks align with diffusion "
        f"blocks. Adjust algo.block_size accordingly.")
    if NUM_COLS % block_size != 0:
      raise ValueError(
        f"NUM_COLS={NUM_COLS} must be a multiple of block_size={block_size}.")
    blocks_per_row = NUM_COLS // block_size
    total_blocks = NUM_ROWS * blocks_per_row

    B = source.shape[0]
    device = self.device
    if condition_grid is not None:
      condition_grid = condition_grid.to(device=device, dtype=torch.long)
      observed_rows = torch.all(condition_grid == UNK_ID, dim=-1)
      observed_content_unk = (
        condition_grid[:, NUM_SOURCE_ROWS:, NUM_PROMPT_COLS:] == UNK_ID)
      if row_unk_mask is None:
        row_unk_mask = observed_rows
      elif not torch.equal(row_unk_mask.bool().to(device), observed_rows):
        raise ValueError(
          'row_unk_mask differs from all-<UNK> rows in condition_grid')
      if content_unk_mask is None:
        content_unk_mask = observed_content_unk
      elif not torch.equal(
          content_unk_mask.bool().to(device), observed_content_unk):
        raise ValueError(
          'content_unk_mask differs from <UNK> cells in condition_grid')
    elif row_unk_mask is None:
      raise ValueError(
        'V2 sampling requires row_unk_mask or an explicit condition_grid')

    row_unk_mask = row_unk_mask.bool().to(device)
    if content_unk_mask is not None:
      content_unk_mask = content_unk_mask.bool().to(device)

    x_accum = self._xmidi_sample_prior(
      source, target_prompts, row_unk_mask,
      content_unk_mask=content_unk_mask,
      condition_grid=condition_grid,
      generation_mask=generation_mask)
    initial_state = x_accum.clone()
    effective_generation_mask = initial_state == self.mask_index

    use_kv_cache = bool(self.config.sampling.kv_cache)
    if use_kv_cache:
      self.backbone.reset_kv_cache(eval_batch_size=B)
    elif hasattr(self.backbone, 'reset_sampling_audit'):
      self.backbone.reset_sampling_audit()

    # XMIDI: register the per-batch row_unk_mask on the backbone so that
    # ``DIT.forward``'s sampling path can build a per-sample attention
    # mask that drops fully-<UNK> rows (R&D §"局部 attention 算法").
    # Cleared in the finally block so other code paths (training,
    # validation across loops) don't accidentally inherit it.
    set_row_unk = (row_unk_mask is not None
                   and hasattr(self.backbone, 'set_xmidi_row_unk_mask'))
    if set_row_unk:
      self.backbone.set_xmidi_row_unk_mask(row_unk_mask)

    sampling_steps = 0
    ones = torch.ones((B, 1), dtype=self.dtype, device=device)
    sigma_zero = torch.zeros(B, 1, device=device)

    try:
      pbar = tqdm(range(total_blocks), desc='xmidi blocks', dynamic_ncols=True)
      for block_idx in pbar:
        row = block_idx // blocks_per_row
        col_start = (block_idx % blocks_per_row) * block_size
        start = row * NUM_COLS + col_start
        end = start + block_size

        is_source_row = (row < NUM_SOURCE_ROWS)
        is_prompt_block = (col_start < NUM_PROMPT_COLS)
        all_unk_for_row = (
          row_unk_mask is not None and bool(row_unk_mask[:, row].all().item())
        )
        # A content block can consist entirely of structural <UNK> cells even
        # when its row is not globally inactive. It still must be committed.
        has_generation = bool(
          (x_accum[:, start:end] == self.mask_index).any().item())
        is_known_block = (
          is_source_row or is_prompt_block or all_unk_for_row
          or not has_generation)

        if is_known_block:
          # Pre-fill: write the (already correct) tokens to the KV-cache. With
          # kv_cache=False there is nothing to do — the tokens already sit in
          # x_accum and the next forward will see them through full-context
          # attention.
          if use_kv_cache:
            self._xmidi_commit_kv_block(
              x_accum[:, start:end], sigma_zero)
            expected_cache = end
            if self.backbone.kv_cache_length() != expected_cache:
              raise RuntimeError(
                f'Cache length after block {block_idx} is '
                f'{self.backbone.kv_cache_length()}, expected {expected_cache}')
          continue

        # ---- target content block: standard BD3-LM block diffusion ----
        dt = 1 / num_steps
        p_x0_cache = None
        timesteps = torch.linspace(1.0, dt, num_steps, device=device)
        t_per_sample = torch.ones((B, 1), dtype=self.dtype, device=device)
        for i in range(num_steps):
          if self.mask_index not in x_accum[:, start:end]:
            break

          if self.config.sampling.first_hitting:
            num_masked = (
              x_accum[:, start:end] == self.mask_index
            ).sum(-1, keepdim=True)
            active = num_masked > 0
            u = torch.rand_like(t_per_sample).clamp_min(1e-12)
            proposal = t_per_sample * u.pow(
              1.0 / num_masked.clamp_min(1).to(t_per_sample.dtype))
            t_per_sample = torch.where(active, proposal, t_per_sample)
            step_t = t_per_sample
          else:
            step_t = timesteps[i] * ones

          p_x0_cache, x_next = self._ddpm_caching_update(
            x=x_accum[:, :end],
            t=step_t,
            dt=dt,
            p_x0=p_x0_cache,
          )
          if p_x0_cache is None:
            sampling_steps += 1
          x_accum[:, :end] = x_next

        if (x_accum[:, start:end] == self.mask_index).any():
          unresolved = int(
            (x_accum[:, start:end] == self.mask_index).sum().item())
          raise RuntimeError(
            f'Block {block_idx} retains {unresolved} <MASK> tokens after '
            f'{num_steps} reverse steps')
        if use_kv_cache and self.backbone.kv_cache_length() != end:
          raise RuntimeError(
            f'Generated block {block_idx} was not committed exactly once: '
            f'cache={self.backbone.kv_cache_length()}, expected={end}')

        # Per-sample UNK rows: if a sample's `row` is fully <UNK> we forced it
        # to UNK in the prior; the diffusion above wrote into a slice that
        # spans every sample, so we restore the contract by overwriting those
        # samples' content back to UNK_ID.
        if row_unk_mask is not None:
          unk_per_sample = row_unk_mask[:, row]
          if unk_per_sample.any():
            x_accum[unk_per_sample, start:end] = UNK_ID

        # R&D §"特殊符号": per-token <UNK> positions inside the content
        # region must remain <UNK> for the whole diffusion chain. The
        # diffusion update above may have overwritten them with
        # generated tokens; restore them here. ``content_unk_mask`` is
        # indexed in (target-row, content-col) space, so we map the
        # current (row, col_start..col_start+block_size) slice into it.
        if (content_unk_mask is not None
            and row >= NUM_SOURCE_ROWS
            and col_start >= NUM_PROMPT_COLS):
          tr = row - NUM_SOURCE_ROWS              # 0..NUM_TARGET_ROWS-1
          tc_lo = col_start - NUM_PROMPT_COLS     # 0..(content_cols-block_size)
          tc_hi = tc_lo + block_size
          # mask_block: (B, block_size) bool
          mask_block = content_unk_mask[:, tr, tc_lo:tc_hi]
          if mask_block.any():
            # Build a flat (B, block_size) boolean indexer aligned with
            # x_accum[:, start:end]; assign via masked scatter.
            content_slice = x_accum[:, start:end]
            content_slice[mask_block] = UNK_ID
            x_accum[:, start:end] = content_slice
    finally:
      if set_row_unk:
        self.backbone.set_xmidi_row_unk_mask(None)

    runtime_trace = (
      self.backbone.get_sampling_audit()
      if hasattr(self.backbone, 'get_sampling_audit') else {})
    if use_kv_cache and self.backbone.kv_cache_length() != x_accum.shape[1]:
      runtime_trace['cache_consistent'] = False
    audits = _xm.audit_xmidi_batch(
      output_ids=x_accum,
      initial_ids=initial_state,
      generation_mask=effective_generation_mask,
      row_unk_mask=row_unk_mask,
      runtime_trace=runtime_trace)
    self._last_xmidi_audit = audits
    failures = [record for record in audits if not record['all_pass']]
    if failures:
      raise RuntimeError(
        f'V2 structural audit failed for {len(failures)}/{len(audits)} '
        f'samples: {failures[0]["failures"]}')

    return x_accum, sampling_steps

  @torch.no_grad()
  def _xmidi_sample_full(self, num_steps, batch_size_per_gpu=None):
    """End-to-end XMIDI sampler: pull source/prompts from the validation
    loader and decode target content for ``num_sample_batches`` batches.

    Returns:
      decoded: list of strings (length B * num_sample_batches), one per
               generated sample, produced by ``self.tokenizer.batch_decode``.
      raw_ids: list of LongTensor, one per batch, [B, 8192] each — the
               full V2 grid (source + prompts + generated content).
    """
    from xmidi_compat import get_xmidi_module
    _xm = get_xmidi_module(self.config)
    NUM_ROWS = _xm.NUM_ROWS
    NUM_COLS = _xm.NUM_COLS
    NUM_SOURCE_ROWS = _xm.NUM_SOURCE_ROWS
    NUM_PROMPT_COLS = _xm.NUM_PROMPT_COLS
    PAD_ID = _xm.PAD_ID
    _, valid_loader = dataloader.get_dataloaders(
      self.config, self.tokenizer, skip_train=True,
      valid_seed=None)

    samples_ids = []
    decoded = []
    nfes_acc = 0
    num_batches = self.config.sampling.num_sample_batches
    it = iter(valid_loader)
    for batch_i in range(num_batches):
      try:
        batch = next(it)
      except StopIteration:
        break
      ids = batch['input_ids'].to(self.device)
      row_unk_mask = batch.get('row_unk_mask', None)
      if row_unk_mask is not None:
        row_unk_mask = row_unk_mask.to(self.device).bool()

      B = ids.shape[0]
      grid = ids.view(B, NUM_ROWS, NUM_COLS)
      source = grid[:, :NUM_SOURCE_ROWS].clone()
      target_prompts = grid[:, NUM_SOURCE_ROWS:, :NUM_PROMPT_COLS].clone()
      # R&D §"特殊符号": every per-token <UNK> position inside the
      # target-content region must stay at <UNK> for the whole diffusion
      # chain. We extract a bool mask over (target_rows, content_cols)
      # straight from the validation GT; the sampler uses it to pre-fill
      # the prior and to re-assert the constraint after each block.
      from xmidi_compat import get_xmidi_module as _gxm
      _UNK = _gxm(self.config).UNK_ID
      content_unk_mask = (
        grid[:, NUM_SOURCE_ROWS:, NUM_PROMPT_COLS:] == _UNK)
      generation_mask = torch.zeros_like(grid, dtype=torch.bool)
      generation_mask[:, NUM_SOURCE_ROWS:, NUM_PROMPT_COLS:] = (
        ~content_unk_mask)
      condition_grid = grid.clone()
      condition_grid[generation_mask] = PAD_ID

      sample_i, nfes = self._xmidi_semi_ar_sampler(
        source, target_prompts, row_unk_mask, num_steps,
        content_unk_mask=content_unk_mask,
        condition_grid=condition_grid,
        generation_mask=generation_mask)
      samples_ids.append(sample_i.detach().cpu())
      decoded.extend(self.tokenizer.batch_decode(sample_i))

      self.metrics.nfes.update(nfes)
      self.metrics.gen_nfes.append(nfes)
      nfes_acc += nfes

    if not samples_ids:
      raise RuntimeError(
        'XMIDI sampler: validation dataloader yielded no batches.')
    return decoded, samples_ids

  def _compute_entropy(self, x):
    _, counts = torch.unique(x, return_counts=True, sorted=False)
    entropy = torch.special.entr(counts.float() / counts.sum()).sum()
    return entropy
  
  def _check_stop_conds(self, x):
    """Check if sampling should stop based on 1) eos, 2) entropy, or 3) likelihood.
    Entropy/likelihood evaluated on last 256 token-block.
    
    Args:
      x: torch.Tensor, current sample.
    Returns:
      stop: bool, whether to stop sampling.
      x: torch.Tensor, sample (potentially truncated for variable-length sampling).
    """
    stop = False # stop sampling?
    truncate_idx = None # truncate sample? (variable-length sampling only)

    # CRITERION 2: always stop sampling if entropy is low
    entropy = self._compute_entropy(x[:, -256:])
    if entropy < 4:
      stop = True

    # for variable length sampling, check if we should stop
    # sampling, and where to truncate the sample
    if self.config.sampling.var_length:
      # CRITERION 1: stop at sampled EOS token
      if len(torch.where(x == self.tokenizer.eos_token_id)[0]) > 1:
        stop = True
        eos_idx = torch.where(x == self.tokenizer.eos_token_id)
        if len(eos_idx[0]) > 1:
          truncate_idx = min(eos_idx[1][1]+1, x.shape[1])

      # CRITERION 2: stop if entropy/likelihood is low
      if entropy < 4:
        stop = True
        truncate_idx = x.shape[1] - 256

    # truncate sample (variable-length sampling only)
    if truncate_idx is not None:
      x = x[:, :truncate_idx]
      if x.ndim == 1:
        x = x.unsqueeze(0)

    return stop, x
