import torchaudio
import torch
import torch.nn as nn
import os
import random
import unicodedata

# ---------------------------------------------------------------------------
# Vietnamese text ↔ integer mapping
# ---------------------------------------------------------------------------

VI_ALPHABET = (
    "aàảãáạăằẳẵắặâầẩẫấậbcdđeèẻẽéẹêềểễếệghiìỉĩíị"
    "klmnoòỏõóọôồổỗốộơờởỡớợpqrstuùủũúụưừửữứựvxyỳỷỹýỵ"
    " '"
)


def normalize_transcript(text: str) -> str:
    """Lowercase, NFC normalize, remove special chars; preserve Vietnamese diacritics."""
    text = unicodedata.normalize("NFC", text.lower().strip())
    return "".join(c for c in text if c in VI_ALPHABET)


class TextTransform:
    """Map Vietnamese characters to integers and vice versa."""

    def __init__(self):
        self.char_map = {ch: i for i, ch in enumerate(VI_ALPHABET)}
        self.index_map = {i: ch for ch, i in self.char_map.items()}
        self.blank_idx = len(VI_ALPHABET)

    def text_to_int(self, text: str):
        text = normalize_transcript(text)
        return [self.char_map[c] for c in text if c in self.char_map]

    def int_to_text(self, labels) -> str:
        return "".join(
            self.index_map[i] for i in labels
            if i != self.blank_idx and i in self.index_map
        )


TEXT_TRANSFORM = TextTransform()
VOCAB_SIZE = len(TEXT_TRANSFORM.char_map)
BLANK_IDX = TEXT_TRANSFORM.blank_idx
NUM_CLASSES = BLANK_IDX + 1


# ---------------------------------------------------------------------------
# Audio feature transforms
# ---------------------------------------------------------------------------

global_mean =  torch.tensor([-24.2660, -18.6908, -13.7669, -14.2111,  -7.1560,  -7.6943,  -7.1851,
         -4.2978,  -4.8221,  -4.1201,  -5.1094,  -5.2236,  -5.1167,  -4.5937,
         -4.6483,  -5.0046,  -5.5109,  -6.1044,  -6.7666,  -7.4232,  -8.0957,
         -8.8795,  -9.7729, -10.7417, -11.8011, -12.7625, -13.2616, -15.1400,
        -16.0306, -15.6396, -17.6837, -17.2780, -18.2830, -18.0426, -18.7767,
        -18.5980, -18.9698, -19.3400, -19.7190, -20.2227, -20.7465, -21.2466,
        -21.5844, -21.8003, -21.8738, -21.7992, -21.3127, -21.4595, -21.1526,
        -21.6093, -21.8270, -22.6595, -23.2346, -23.8137, -24.2570, -24.5193,
        -24.7054, -24.8437, -25.3006, -25.9214, -26.5420, -27.1251, -27.7053,
        -28.1566, -28.6208, -29.0741, -29.5604, -29.9314, -30.2777, -30.7121,
        -31.2983, -32.1564, -32.5009, -32.7972, -32.9734, -33.4359, -34.1482,
        -35.1747, -36.3483, -37.5019])
global_std =  torch.tensor([14.1405, 14.1811, 13.9281, 13.9205, 14.1637, 13.2881, 13.7293, 14.0519,
        13.8300, 13.7818, 14.4067, 14.3432, 14.0448, 14.1427, 14.3695, 14.5572,
        14.6690, 14.7115, 14.7504, 14.7723, 14.7598, 14.6666, 14.4875, 14.2403,
        13.9664, 13.7077, 13.4428, 13.2807, 13.1455, 12.9886, 12.9686, 12.8391,
        12.8301, 12.8190, 12.8329, 12.8502, 12.8490, 12.7844, 12.7240, 12.6706,
        12.6346, 12.6049, 12.5822, 12.5867, 12.6105, 12.6370, 12.6406, 12.6733,
        12.6741, 12.6826, 12.6401, 12.5759, 12.4882, 12.4318, 12.3985, 12.3978,
        12.4214, 12.4265, 12.3996, 12.3274, 12.2363, 12.1644, 12.0919, 11.9859,
        11.8506, 11.7616, 11.7003, 11.7108, 11.7595, 11.7753, 11.7273, 11.6405,
        11.5417, 11.4557, 11.3879, 11.3333, 11.2739, 11.2438, 11.1666, 11.0244])

class GlobalCMVN(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, x):
        mean = self.mean.view(-1, 1)
        std = self.std.view(-1, 1)
        return (x - mean) / (std + 1e-5)

def get_audio_transforms():
  
  #  10 time masks with p=0.05
  #  The actual conformer paper uses a variable time_mask_param based on the length of each utterance.
  #  For simplicity, we approximate it with just a fixed value.
  time_masks = [torchaudio.transforms.TimeMasking(time_mask_param=80, p=0.05) for _ in range(3)]
  train_audio_transform = nn.Sequential(torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80, hop_length=160),
                                        torchaudio.transforms.AmplitudeToDB(), #80 filter banks, 25ms window size, 10ms hop
                                        GlobalCMVN(global_mean, global_std),
                                        torchaudio.transforms.FrequencyMasking(freq_mask_param=27),
                                        torchaudio.transforms.FrequencyMasking(27),
                                        *time_masks,
  )

  valid_audio_transform = nn.Sequential(
      torchaudio.transforms.MelSpectrogram(
          sample_rate=16000,
          n_mels=80,
          hop_length=160,
      ),
      torchaudio.transforms.AmplitudeToDB(),
      GlobalCMVN(global_mean, global_std))
  
  return train_audio_transform, valid_audio_transform

# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

class BatchSampler(object):
    ''' Sample contiguous, sorted indices. Leads to less padding and faster training. '''

    def __init__(self, sorted_inds, batch_size):
        self.sorted_inds = sorted_inds
        self.batch_size = batch_size

    def __iter__(self):
        inds = self.sorted_inds.copy()
        while len(inds):
            to_take = min(self.batch_size, len(inds))
            start_ind = random.randint(0, len(inds) - to_take)
            batch_inds = inds[start_ind:start_ind + to_take]
            del inds[start_ind:start_ind + to_take]
            yield batch_inds

    def __len__(self):
        return (len(self.sorted_inds) + self.batch_size - 1) // self.batch_size

# ---------------------------------------------------------------------------
# Collate / preprocessing
# ---------------------------------------------------------------------------

def preprocess_example(data, data_type="train", text_transform=None):
  ''' Process raw LibriSpeech examples '''
  text_transform = text_transform or TEXT_TRANSFORM
  train_audio_transform, valid_audio_transform = get_audio_transforms()
  spectrograms = []
  labels = []
  references = []
  input_lengths = []
  label_lengths = []
  for (waveform, _, utterance) in data:
    # Generate spectrogram for model input
    if data_type == 'train':
      spec = train_audio_transform(waveform).squeeze(0).transpose(0, 1) # (1, time, freq)
    else:
      spec = valid_audio_transform(waveform).squeeze(0).transpose(0, 1) # (1, time, freq)
    spectrograms.append(spec)

    # Labels 
    utterance = normalize_transcript(utterance)
    references.append(utterance) # Actual Sentence
    # label = torch.Tensor(text_transform.text_to_int(utterance)) # Integer representation of sentence
    label = torch.tensor(
    text_transform.text_to_int(utterance),
    dtype=torch.long
    )
    labels.append(label)

    # Lengths (time)
    input_lengths.append(((spec.shape[0] - 1) // 2 - 1) // 2) # account for subsampling of time dimension
    label_lengths.append(len(label))

  # Pad batch to length of longest sample
  spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
  labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

  # Padding mask (batch_size, time, time)
  mask = torch.ones(spectrograms.shape[0], spectrograms.shape[1], spectrograms.shape[1])
  for i, l in enumerate(input_lengths):
    mask[i, :, :l] = 0

  return spectrograms, labels, input_lengths, label_lengths, references, mask.bool()


def collate_cached_features(batch):
  '''Collate function for datasets that already store spectrograms.'''
  spectrograms = [item[0].to(torch.float32) for item in batch]
  labels = [item[1] for item in batch]
  input_lengths = [item[2] for item in batch]
  label_lengths = [item[3] for item in batch]
  references = [item[4] for item in batch]

  spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
  labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

  mask = torch.ones(spectrograms.shape[0], spectrograms.shape[1], spectrograms.shape[1])
  for i, l in enumerate(input_lengths):
    mask[i, :, :l] = 0

  return spectrograms, labels, input_lengths, label_lengths, references, mask.bool()

# ---------------------------------------------------------------------------
# LR scheduler
# ---------------------------------------------------------------------------

class TransformerLrScheduler():
  '''
    Transformer LR scheduler from "Attention is all you need." https://arxiv.org/abs/1706.03762
    multiplier and warmup_steps taken from conformer paper: https://arxiv.org/abs/2005.08100
  '''
  def __init__(self, optimizer, d_model, warmup_steps, multiplier=5):
    self._optimizer = optimizer
    self.d_model = d_model
    self.warmup_steps = warmup_steps
    self.n_steps = 0
    self.multiplier = multiplier

  def step(self):
    self.n_steps += 1
    lr = self._get_lr()
    for param_group in self._optimizer.param_groups:
        param_group['lr'] = lr

  def _get_lr(self):
    return self.multiplier * (self.d_model ** -0.5) * min(self.n_steps ** (-0.5), self.n_steps * (self.warmup_steps ** (-1.5)))

# ---------------------------------------------------------------------------
# Misc utilities
# ---------------------------------------------------------------------------

class AvgMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = self.sum = None
        self.cnt = 0

    def update(self, val, n=1):
        self.sum = val * n if self.sum is None else self.sum + val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class GreedyCharacterDecoder(nn.Module):
    def __init__(self, blank_idx=BLANK_IDX):
        super().__init__()
        self.blank_idx = blank_idx

    def forward(self, x, lengths=None):
        indices = torch.argmax(x, dim=-1)
        if lengths is not None:
            if not torch.is_tensor(lengths):
                lengths = torch.tensor(lengths, device=indices.device, dtype=torch.long)
            max_len = indices.size(1)
            valid = torch.arange(max_len, device=indices.device).unsqueeze(0) < lengths.unsqueeze(1)
            indices = torch.where(valid, indices, torch.full_like(indices, self.blank_idx))
        indices = torch.unique_consecutive(indices, dim=-1)
        return [
            [t for t in row if t != self.blank_idx]
            for row in indices.tolist()
        ]


class CTCBeamSearchDecoder(nn.Module):
    """CTC beam search decoder for inference. Greedy when beam_size=1."""

    def __init__(self, blank_idx=BLANK_IDX, beam_size: int = 10):
        super().__init__()
        self.blank_idx = blank_idx
        self.beam_size = beam_size

    def _beam_search_one(self, log_probs, length):
        T, C = log_probs.shape
        beam = {(): 0.0}
        for t in range(min(length, T)):
            new_beam = {}
            for seq, score in beam.items():
                for c in range(C):
                    lp = log_probs[t, c].item()
                    if c == self.blank_idx:
                        new_seq, ns = seq, score + lp
                    else:
                        new_seq = seq + (c,) if (not seq or seq[-1] != c) else seq
                        ns = score + lp
                    new_beam[new_seq] = max(new_beam.get(new_seq, -1e9), ns)
            beam = dict(sorted(new_beam.items(), key=lambda x: -x[1])[: self.beam_size])
        best = max(beam.items(), key=lambda x: x[1])
        return [t for t in best[0] if t != self.blank_idx]

    def forward(self, logits, lengths=None):
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        if lengths is not None and not torch.is_tensor(lengths):
            lengths = torch.tensor(lengths, device=logits.device, dtype=torch.long)
        B = logits.size(0)
        if self.beam_size <= 1:
            dec = GreedyCharacterDecoder(self.blank_idx)
            return dec(logits, lengths)
        lens = lengths.tolist() if torch.is_tensor(lengths) else (lengths or [log_probs.size(1)] * B)
        return [self._beam_search_one(log_probs[i, :l].cpu(), l) for i, l in enumerate(lens)]


def model_size(model, name):
    params = sum(p.nelement() for p in model.parameters())
    size_mb = sum(p.nelement() * p.element_size() for p in model.parameters()) / 1024 ** 2
    print(f"{name} — params: {params / 1e6:.2f}M,  size: {size_mb:.2f} MB")

def add_model_noise(model, std=0.0001, gpu=True):
  '''
    Add variational noise to model weights: https://ieeexplore.ieee.org/abstract/document/548170
    STD may need some fine tuning...
  '''
  with torch.no_grad():
    for param in model.parameters():
        if gpu:
          param.add_(torch.randn(param.size()).cuda() * std)
        else:
          param.add_(torch.randn(param.size()).cuda() * std)

# ---------------------------------------------------------------------------
# Checkpoint helpers (encoder + decoder pattern)
# ---------------------------------------------------------------------------

def load_checkpoint(encoder, decoder, optimizer, scheduler, checkpoint_path):
  ''' Load model checkpoint '''
  if not os.path.exists(checkpoint_path):
    raise 'Checkpoint does not exist'
  checkpoint = torch.load(checkpoint_path)
  scheduler.n_steps = checkpoint['scheduler_n_steps']
  scheduler.multiplier = checkpoint['scheduler_multiplier']
  scheduler.warmup_steps = checkpoint['scheduler_warmup_steps']
  encoder.load_state_dict(checkpoint['encoder_state_dict'])
  decoder.load_state_dict(checkpoint['decoder_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  return checkpoint['epoch'], checkpoint['valid_loss']

def save_checkpoint(encoder, decoder, optimizer, scheduler, valid_loss, epoch, checkpoint_path):
  ''' Save model checkpoint '''
  torch.save({
            'epoch': epoch,
            'valid_loss': valid_loss,
            'scheduler_n_steps': scheduler.n_steps,
            'scheduler_multiplier': scheduler.multiplier,
            'scheduler_warmup_steps': scheduler.warmup_steps,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
