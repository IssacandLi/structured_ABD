import os
import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch
import transformers

import dataloader
import diffusion
import utils

omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)


def _load_from_checkpoint(config, tokenizer):
  if 'hf' in config.algo.backbone:
    return diffusion.Diffusion(
      config, tokenizer=tokenizer).to('cuda')
  
  return diffusion.Diffusion.load_from_checkpoint(
    config.eval.checkpoint_path,
    tokenizer=tokenizer,
    config=config,
    strict=False,
    weights_only=False).to('cuda')

@L.pytorch.utilities.rank_zero_only
def _print_config(
  config: omegaconf.DictConfig,
  resolve: bool = True,
  save_cfg: bool = True) -> None:
  """Prints content of DictConfig using Rich library and its tree structure.
  
  Args:
    config (DictConfig): Configuration composed by Hydra.
    resolve (bool): Whether to resolve reference fields of DictConfig.
    save_cfg (bool): Whether to save the configuration tree to a file.
  """

  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

  fields = config.keys()
  for field in fields:
    branch = tree.add(field, style=style, guide_style=style)

    config_section = config.get(field)
    branch_content = str(config_section)
    if isinstance(config_section, omegaconf.DictConfig):
      branch_content = omegaconf.OmegaConf.to_yaml(
        config_section, resolve=resolve)

    branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
  rich.print(tree)
  if save_cfg:
    with fsspec.open(
      '{}/config_tree.txt'.format(
        config.checkpointing.save_dir), 'w') as fp:
      rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
  for dl_type, dl in [
    ('train', train_ds), ('valid', valid_ds)]:
    #==========disable validation=========
    if dl is None:
      print(f'Skipping {dl_type} dataloader batch (None).')
      continue
    print(f'Printing {dl_type} dataloader batch.')
    print(f'Printing {dl_type} dataloader batch.')

    batch = next(iter(dl))
    print('Batch input_ids.shape', batch['input_ids'].shape)
    first = batch['input_ids'][0, :k]
    last = batch['input_ids'][0, -k:]
    print(f'First {k} tokens:', tokenizer.decode(first))
    print('ids:', first)
    print(f'Last {k} tokens:', tokenizer.decode(last))
    print('ids:', last)

def generate_samples(config, logger, tokenizer):
  conditional = bool(getattr(config.data, "conditional_generation", False))
  logger.info('Generating samples.')
  model = _load_from_checkpoint(config=config,
                                tokenizer=tokenizer)
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None

  if conditional:
    logger.info('Conditional sampling: using dataloader prefixes and generating answer tokens.')
    _, valid_dl = dataloader.get_dataloaders(
      config, tokenizer, skip_train=True, valid_seed=config.seed)

    preds = []
    prefixes = []
    gts = []

    num_batches = int(getattr(config.sampling, "num_cond_batches", 1))
    num_batches = max(1, num_batches)

    model.backbone.eval()
    for b_idx, batch in enumerate(valid_dl):
      if b_idx >= num_batches:
        break
      # Move batch tensors to CPU/GPU handled inside model; keep here light.
      # Generate answer-only text (return_full_sequence=False)
      pred_i = model.restore_model_and_sample_conditional(
        num_steps=config.algo.T,
        batch=batch,
        token_mask_key='attention_mask',
        return_full_sequence=False)

      # Extract prefix text + ground-truth answer from batch for logging
      input_ids = batch['input_ids']
      mask = batch['attention_mask']
      pad_id = tokenizer.pad_token_id

      for i in range(input_ids.shape[0]):
        ids = input_ids[i].tolist()
        m = mask[i].tolist()

        # prefix: up to first answer token (first m==1), ignoring PAD
        try:
          ans_start = m.index(1)
        except ValueError:
          ans_start = len(ids)
        prefix_ids = [t for t in ids[:ans_start] if t != pad_id]
        gt_ids = [t for t, mm in zip(ids, m) if mm == 1 and t != pad_id]

        # cut GT at first EOS if present
        if tokenizer.eos_token_id in gt_ids:
          gt_ids = gt_ids[:gt_ids.index(tokenizer.eos_token_id)]

        prefixes.append(tokenizer.decode(prefix_ids))
        gts.append(tokenizer.decode(gt_ids))

      # pred_i is a list length = batch size
      preds.extend(pred_i)
    model.metrics.reset()  # 可选：避免遗留值；更保险
    model.metrics.record_conditional_perplexity(
        prefixes=prefixes,
        answers=preds,          # 这里用你的生成 answer 文本
        max_length=2048,
        device='cuda',
    )
    print("Generative perplexity:", model.metrics.gen_ppl.compute())
    # Print a few examples
    k = min(3, len(preds))
    for i in range(k):
      print('\n=== Example', i, '===')
      print('PREFIX:\n', prefixes[i])
      print('GT:\n', gts[i])
      print('PRED:\n', preds[i])

    csv_path = config.sampling.logdir
    save_dict = {
      'prefix': [[p] for p in prefixes],
      'gt': [[g] for g in gts],
      'pred': [[p] for p in preds],
      'seed': [config.seed for _ in range(len(preds))],
      'gen_ppl': model.metrics.gen_ppls,
    }

    utils.update_and_save_csv(save_dict, csv_path)
    return preds


  text_samples = model.restore_model_and_sample(
    num_steps=config.algo.T)
  print('Text samples:', text_samples)
  print('Generative perplexity:',
        model.metrics.gen_ppl.compute())
  print('Entropy:', model.metrics.gen_entropy.compute())
  csv_path = config.sampling.logdir
  save_dict = {'gen_ppl': model.metrics.gen_ppls,
                'gen_nfes': model.metrics.gen_nfes,
                'gen_entropy': model.metrics.gen_entropies,
                'gen_lengths': model.metrics.gen_lengths,
                'samples': [[i] for i in text_samples],
                'seed': [config.seed for _ in range(len(text_samples))]}
  if config.sampling.var_length:
    print(text_samples)
    save_dict['samples'] = ['' for _ in range(len(text_samples))]
  utils.update_and_save_csv(save_dict, csv_path)
  return text_samples

def _ppl_eval(config, logger, tokenizer):
  logger.info('Starting Eval.')
  model = _load_from_checkpoint(config=config,
                                tokenizer=tokenizer)

  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None

  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))
  seed = config.seed
  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)
  L.seed_everything(seed)
  config.seed = seed
  _, valid_ds = dataloader.get_dataloaders(
    config, tokenizer, skip_train=True, valid_seed=seed)
  trainer.validate(model, valid_ds)

def _train(config, logger, tokenizer):
  logger.info('Starting Training.')
  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)

  if (config.checkpointing.resume_from_ckpt
      and config.checkpointing.resume_ckpt_path is not None
      and utils.fsspec_exists(
        config.checkpointing.resume_ckpt_path)):
    ckpt_path = config.checkpointing.resume_ckpt_path
    logger.info(f'Resuming training at {ckpt_path}')
  else:
    ckpt_path = None

  # Lightning callbacks
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))

  # train_ds, valid_ds = dataloader.get_dataloaders(
  #   config, tokenizer)
  # _print_batch(train_ds, valid_ds, tokenizer)

  #-=========disable validation====
  disable_val = bool(config.training.get("disable_validation", False))

  if disable_val:
    train_ds, _ = dataloader.get_dataloaders(config, tokenizer, skip_valid=True)
    valid_ds = None
  else:
    train_ds, valid_ds = dataloader.get_dataloaders(config, tokenizer)

  _print_batch(train_ds, valid_ds, tokenizer)

  if config.training.from_pretrained is not None and ckpt_path is None:
    logger.info(f'Loading pretrained model from {config.training.from_pretrained}')
    # load pretraining checkpoint
    if 'kuleshov-group/' in config.training.from_pretrained:
      # load from hf
      model = diffusion.Diffusion(config, tokenizer=tokenizer)
      state_dict = transformers.AutoModelForMaskedLM.from_pretrained(
          config.training.from_pretrained,
          trust_remote_code=True
      ).state_dict()
      model.load_state_dict(state_dict)
    else:
      model = diffusion.Diffusion.load_from_checkpoint(
        config.training.from_pretrained,
        tokenizer=tokenizer,
        config=config,
        strict=False)
    # add buffers for grid search
    model.register_buffer('sampling_eps_min', torch.tensor(
      config.training.sampling_eps_min))
    model.register_buffer('sampling_eps_max', torch.tensor(
      config.training.sampling_eps_max))
  else:
    logger.info(f'Initializing new model')
    model = diffusion.Diffusion(
      config, tokenizer=valid_ds.tokenizer)
  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)

  # trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)
  #if disable validation=================

  if disable_val:
    trainer.fit(model, train_ds, ckpt_path=ckpt_path)
  else:
    trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)
  
@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
  """Main entry point for training."""
  L.seed_everything(config.seed)
  _print_config(config, resolve=True, save_cfg=True)
  
  logger = utils.get_logger(__name__)
  tokenizer = dataloader.get_tokenizer(config)

  if config.mode == 'sample_eval':
    config.wandb = None
    samples = generate_samples(config, logger, tokenizer)
  elif config.mode == 'ppl_eval':
    config.wandb = None
    _ppl_eval(config, logger, tokenizer)
  else:
    _train(config, logger, tokenizer)


if __name__ == '__main__':
  main()