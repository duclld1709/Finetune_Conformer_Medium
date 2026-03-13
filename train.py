import os
import gc
import argparse
import torchaudio
import torch
import torch.nn.functional as F
from functools import partial
from tqdm.auto import tqdm
import wandb

from torch import nn
from torchmetrics.text.wer import WordErrorRate
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from model import ConformerEncoder, LSTMDecoder
from utils import *
from dataset import ViMD, CachedFeatureDataset

parser = argparse.ArgumentParser("conformer")
parser.add_argument('--data_dir', type=str, default='./data', help='location to download data')
parser.add_argument('--checkpoint_path', type=str, default='model_best.pt', help='path to store/load checkpoints')
parser.add_argument('--load_checkpoint', action='store_true', default=False, help='resume training from checkpoint')
parser.add_argument('--train_set', type=str, default='train-clean-100', help='train dataset')
parser.add_argument('--test_set', type=str, default='test-clean', help='test dataset')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--warmup_steps', type=float, default=10000, help='Multiply by sqrt(d_model) to get max_lr')
parser.add_argument('--peak_lr_ratio', type=int, default=0.05, help='Number of warmup steps for LR scheduler')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id (optional)')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--report_freq', type=int, default=100, help='training objective report frequency')
parser.add_argument('--layers', type=int, default=8, help='total number of layers') 
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--use_amp', action='store_true', default=False, help='use mixed precision to train')
parser.add_argument('--attention_heads', type=int, default=4, help='number of heads to use for multi-head attention')
parser.add_argument('--d_input', type=int, default=80, help='dimension of the input (num filter banks)')
parser.add_argument('--d_encoder', type=int, default=256, help='dimension of the encoder')
parser.add_argument('--d_decoder', type=int, default=640, help='dimension of the decoder')
parser.add_argument('--encoder_layers', type=int, default=16, help='number of conformer blocks in the encoder')
parser.add_argument('--decoder_layers', type=int, default=1, help='number of decoder layers')
parser.add_argument('--conv_kernel_size', type=int, default=32, help='size of kernel for conformer convolution blocks')
parser.add_argument('--feed_forward_expansion_factor', type=int, default=4, help='expansion factor for conformer feed forward blocks')
parser.add_argument('--feed_forward_residual_factor', type=int, default=.5, help='residual factor for conformer feed forward blocks')
parser.add_argument('--dropout', type=float, default=.1, help='dropout factor for conformer model')
parser.add_argument('--weight_decay', type=float, default=1e-6, help='model weight decay (corresponds to L2 regularization)')
parser.add_argument('--variational_noise_std', type=float, default=.0001, help='std of noise added to model weights for regularization')
parser.add_argument('--num_workers', type=int, default=0, help='num_workers for the dataloader')
parser.add_argument('--smart_batch', type=bool, default=True, help='Use smart batching for faster training')
parser.add_argument('--accumulate_iters', type=int, default=1, help='Number of iterations to accumulate gradients')
parser.add_argument('--feature_cache_dir', type=str, default=None, help='Root directory that stores cached features split into train/test subfolders')
parser.add_argument('--train_feature_cache_dir', type=str, default=None, help='Direct path to cached train split (overrides --feature_cache_dir)')
parser.add_argument('--test_feature_cache_dir', type=str, default=None, help='Direct path to cached test/valid split (overrides --feature_cache_dir)')
parser.add_argument('--feature_cache_dtype', type=str, default='float32', choices=['float32', 'float16', 'float64'], help='Data type to use when loading cached spectrograms')
parser.add_argument("--beam_size", type=int, default=1, help="CTC beam size for eval; 1=greedy")
parser.add_argument('--verbose_val', action='store_true', help='print sample predictions during validation')
parser.add_argument("--wandb_project", type=str, default="conformer-medium")
parser.add_argument("--wandb_run_name", type=str, default=None)
parser.add_argument("--wandb_entity", type=str, default=None)
parser.add_argument("--wandb_disabled", action="store_true", default=False)
args = parser.parse_args()

def _resolve_cache_dir(split):
  if split == 'train' and args.train_feature_cache_dir:
    return args.train_feature_cache_dir, None
  if split == 'test' and args.test_feature_cache_dir:
    return args.test_feature_cache_dir, None
  if args.feature_cache_dir:
    return args.feature_cache_dir, split
  return None, None


def _get_sorted_indices(dataset):
  if hasattr(dataset, 'frame_lengths'):
    frame_lengths = getattr(dataset, 'frame_lengths')
    if frame_lengths and not any(l is None for l in frame_lengths):
      return sorted(range(len(dataset)), key=lambda idx: frame_lengths[idx])

  lengths = []
  for ind, sample in enumerate(dataset):
    seq = sample[0]
    if seq.ndim > 1 and seq.shape[0] == 1:
      length = seq.shape[1]
    else:
      length = seq.shape[0]
    lengths.append((ind, length))
  return [ind for ind, _ in sorted(lengths, key=lambda x: x[1])]


def get_grad_norm(parameters):
  """Compute the total gradient norm across all parameters."""
  total_norm = 0.0
  for p in parameters:
    if p.grad is not None:
      total_norm += p.grad.detach().float().norm(2).item() ** 2
  return total_norm ** 0.5


def main():

  wandb.init(project=args.wandb_project, 
             name=args.wandb_run_name, 
             entity=args.wandb_entity, 
             config=vars(args), 
             mode="disabled" if args.wandb_disabled else "online")

  # Define wandb metrics so that all charts use global_step as x-axis
  wandb.define_metric("global_step")
  wandb.define_metric("train/*", step_metric="global_step")
  wandb.define_metric("eval/*",  step_metric="global_step")

  # Load Data
  text_transform = TEXT_TRANSFORM
  train_cache_dir, train_split = _resolve_cache_dir('train')
  test_cache_dir, test_split = _resolve_cache_dir('test')

  if train_cache_dir:
    print(f'Using cached features for train split from {os.path.join(train_cache_dir, train_split or "")}')
    train_data = CachedFeatureDataset(train_cache_dir, split=train_split, dtype=args.feature_cache_dtype)
    collate_train = collate_cached_features
  else:
    train_data = ViMD(split='train')
    collate_train = partial(preprocess_example, data_type='train', text_transform=text_transform)

  if test_cache_dir:
    print(f'Using cached features for test split from {os.path.join(test_cache_dir, test_split or "")}')
    test_data = CachedFeatureDataset(test_cache_dir, split=test_split, dtype=args.feature_cache_dtype)
    collate_valid = collate_cached_features
  else:
    test_data = ViMD(split='test')
    collate_valid = partial(preprocess_example, data_type='valid', text_transform=text_transform)
  num_classes = text_transform.blank_idx + 1

  if args.smart_batch:
    print('Sorting training data for smart batching...')
    sorted_train_inds = _get_sorted_indices(train_data)
    sorted_test_inds = _get_sorted_indices(test_data)
    train_loader = DataLoader(dataset=train_data,
                                    pin_memory=True,
                                    num_workers=args.num_workers,
                                    batch_sampler=BatchSampler(sorted_train_inds, batch_size=args.batch_size),
                                    collate_fn=collate_train)

    test_loader = DataLoader(dataset=test_data,
                                pin_memory=True,
                                num_workers=args.num_workers,
                                batch_sampler=BatchSampler(sorted_test_inds, batch_size=args.batch_size),
                                collate_fn=collate_valid)
  else:
    train_loader = DataLoader(dataset=train_data,
                                    pin_memory=True,
                                    num_workers=args.num_workers,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    collate_fn=collate_train)

    test_loader = DataLoader(dataset=test_data,
                                pin_memory=True,
                                num_workers=args.num_workers,
                                batch_size=args.batch_size,
                                shuffle=False,
                                collate_fn=collate_valid)


  # Declare Models  
  
  encoder = ConformerEncoder(
                      d_input=args.d_input,
                      d_model=args.d_encoder,
                      num_layers=args.encoder_layers,
                      conv_kernel_size=args.conv_kernel_size, 
                      dropout=args.dropout,
                      feed_forward_residual_factor=args.feed_forward_residual_factor,
                      feed_forward_expansion_factor=args.feed_forward_expansion_factor,
                      num_heads=args.attention_heads)
  
  decoder = LSTMDecoder(
                  d_encoder=args.d_encoder, 
                  d_decoder=args.d_decoder, 
                  num_layers=args.decoder_layers,
                  num_classes=num_classes)
  char_decoder = GreedyCharacterDecoder(blank_idx=text_transform.blank_idx).eval()
  beam_decoder = CTCBeamSearchDecoder(blank_idx=TEXT_TRANSFORM.blank_idx, beam_size=args.beam_size).eval()
  criterion = nn.CTCLoss(blank=text_transform.blank_idx, zero_infinity=True)
  optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=5e-4, betas=(.9, .98), eps=1e-05 if args.use_amp else 1e-09, weight_decay=args.weight_decay)
  scheduler = TransformerLrScheduler(optimizer, args.d_encoder, args.warmup_steps)

  # Print model size
  model_size(encoder, 'Encoder')
  model_size(decoder, 'Decoder')

  gc.collect()

  # GPU Setup
  if torch.cuda.is_available():
    print('Using GPU')
    gpu = True
    torch.cuda.set_device(args.gpu)
    criterion = criterion.cuda()
    encoder = encoder.cuda()
    decoder = decoder.cuda()
    char_decoder = char_decoder.cuda()
    torch.cuda.empty_cache()
  else:
    gpu = False

  # Mixed Precision Setup
  if args.use_amp:
    print('Using Mixed Precision')
  grad_scaler = GradScaler(enabled=args.use_amp)

  # Initialize Checkpoint 
  if args.load_checkpoint:
    start_epoch, best_loss = load_checkpoint(encoder, decoder, optimizer, scheduler, args.checkpoint_path)
    print(f'Resuming training from checkpoint starting at epoch {start_epoch}.')
  else:
    start_epoch = 0
    best_loss = float('inf')

  # global_step tracks optimizer updates across all epochs
  global_step = start_epoch * len(train_loader) // args.accumulate_iters

  # Train Loop
  optimizer.zero_grad()
  for epoch in range(start_epoch, args.epochs):
    torch.cuda.empty_cache()

    #variational noise for regularization
    add_model_noise(encoder, std=args.variational_noise_std, gpu=gpu)
    add_model_noise(decoder, std=args.variational_noise_std, gpu=gpu)

    # Train/Validation loops
    wer, loss, global_step = train(
        encoder, decoder, char_decoder,
        optimizer, scheduler, criterion, grad_scaler,
        train_loader, text_transform, args,
        gpu=gpu, global_step=global_step
    )

    valid_wer, valid_loss = validate(
        encoder, decoder, beam_decoder,
        criterion, test_loader, text_transform, args,
        gpu=gpu, global_step=global_step
    )

    print(f'Epoch {epoch} - Valid WER: {valid_wer}%, Valid Loss: {valid_loss}, Train WER: {wer}%, Train Loss: {loss}')  

    # Save checkpoint + log best model to wandb
    if valid_loss <= best_loss:
      print('Validation loss improved, saving checkpoint.')
      best_loss = valid_loss
      save_checkpoint(encoder, decoder, optimizer, scheduler, valid_loss, epoch+1, args.checkpoint_path)

      # Log best model artifact to wandb
      artifact = wandb.Artifact(
          name=f"best_model",
          type="model",
          metadata={"epoch": epoch, "valid_loss": valid_loss, "valid_wer": valid_wer}
      )
      artifact.add_file(args.checkpoint_path)
      wandb.log_artifact(artifact, aliases=["best", f"epoch_{epoch}"])

  wandb.finish()


def train(encoder, decoder, char_decoder, optimizer, scheduler, criterion, grad_scaler,
          train_loader, text_transform, args, gpu=True, global_step=0):
  ''' Run a single training epoch '''

  wer = WordErrorRate()
  error_rate = AvgMeter()
  avg_loss = AvgMeter()

  encoder.train()
  decoder.train()
  pbar = tqdm(train_loader, total=len(train_loader), desc="Training", leave=False)

  all_params = list(encoder.parameters()) + list(decoder.parameters())

  for i, batch in enumerate(pbar):
    scheduler.step()
    gc.collect()
    spectrograms, labels, input_lengths, label_lengths, references, mask = batch 
    
    # Move to GPU
    if gpu:
      spectrograms = spectrograms.cuda()
      labels = labels.cuda()
      input_lengths = torch.tensor(input_lengths).cuda()
      label_lengths = torch.tensor(label_lengths).cuda()
      mask = mask.cuda()
    
    # Forward pass
    with autocast('cuda', enabled=args.use_amp):
      outputs = encoder(spectrograms, mask)
      outputs = decoder(outputs)
      loss = criterion(F.log_softmax(outputs, dim=-1).transpose(0, 1), labels, input_lengths, label_lengths)

    grad_scaler.scale(loss).backward()

    if (i + 1) % args.accumulate_iters == 0:
      # Unscale before reading grad norms so values are in true fp32 scale
      grad_scaler.unscale_(optimizer)

      torch.nn.utils.clip_grad_norm_(all_params, 3.0)

      grad_norm = get_grad_norm(all_params)

      grad_scaler.step(optimizer)
      grad_scaler.update()
      optimizer.zero_grad()
      global_step += 1

      # Current learning rate (take from first param group)
      current_lr = optimizer.param_groups[0]['lr']

      # Per-step wandb logging
      wandb.log({
          "train/loss":      loss.detach().item(),
          "train/grad_norm": grad_norm,
          "train/lr":        current_lr,
          "global_step":     global_step,
      }, step=global_step)

    avg_loss.update(loss.detach().item())

    # Predict words, compute WER
    inds = char_decoder(outputs.detach(), lengths=input_lengths)
    predictions = []
    for sample in inds:
      predictions.append(text_transform.int_to_text(sample))
    error_rate.update(wer(predictions, references) * 100)

    # Update progress bar
    if (i+1) % args.report_freq == 0:
      pbar.set_postfix({
          "loss": f"{avg_loss.avg:.4f}",
          "WER": f"{error_rate.avg:.2f}%"
      }) 

    del spectrograms, labels, input_lengths, label_lengths, references, outputs, inds, predictions

  return error_rate.avg, avg_loss.avg, global_step


def validate(encoder, decoder, char_decoder, criterion, test_loader, text_transform, args,
             gpu=True, global_step=0):
  ''' Evaluate model on test dataset. '''

  avg_loss = AvgMeter()
  error_rate = AvgMeter()
  wer = WordErrorRate()

  encoder.eval()
  decoder.eval()

  printed_samples = 0

  for i, batch in enumerate(tqdm(test_loader, total=len(test_loader), desc="Validating", leave=False)):

    spectrograms, labels, input_lengths, label_lengths, references, mask = batch 

    if gpu:
      spectrograms = spectrograms.cuda()
      labels = labels.cuda()
      mask = mask.cuda()

      device = spectrograms.device
      input_lengths = torch.as_tensor(input_lengths, device=device)
      label_lengths = torch.as_tensor(label_lengths, device=device)

    with torch.no_grad():
      with autocast('cuda', enabled=args.use_amp):

        outputs = encoder(spectrograms, mask)
        outputs = decoder(outputs)

        loss = criterion(
            F.log_softmax(outputs, dim=-1).transpose(0, 1),
            labels,
            input_lengths,
            label_lengths
        )

    avg_loss.update(loss.item())

    inds = char_decoder(outputs.detach(), lengths=input_lengths)
    predictions = [text_transform.int_to_text(sample) for sample in inds]

    error_rate.update(wer(predictions, references) * 100)

    if args.verbose_val and printed_samples < 3:
      for ref, pred in zip(references, predictions):

        print(f"\nSample {printed_samples+1}")
        print(f"REF: [{ref}]")
        print(f"HYP: [{pred}]")
        print("-"*50)

        printed_samples += 1
        if printed_samples >= 3:
          break

  # Log eval metrics at the current global_step (end of training epoch)
  wandb.log({
      "eval/loss": avg_loss.avg,
      "eval/wer":  error_rate.avg,
      "global_step": global_step,
  }, step=global_step)

  return error_rate.avg, avg_loss.avg


if __name__ == '__main__':

  main()
