"""Live target speaker extraction: microphone -> UniSE TSE -> speaker.

The model works on a fixed window (5 s by default). Instead of processing
disjoint windows, the window slides forward with the audio: each step re-reads
the latest window but only generates tokens for the audio that arrived since the
last step, feeding the tokens already generated for the overlap back in as
history. Each step consumes whatever audio arrived while the previous step ran,
so the delay stays at roughly two step durations instead of a whole window.

Use headphones: with laptop speakers the mic re-captures the output and the
extracted voice loops back into the model.

Examples:
    python realtime_tse.py --list_devices
    python realtime_tse.py --record_enroll 5            # record the target voice first, then go live
    python realtime_tse.py --enroll my_voice.wav        # reuse a saved reference
    python realtime_tse.py --enroll ref.wav --input_wav mix.wav --output_wav out.wav --no_play   # offline simulation
"""
import argparse
import pathlib
import queue
import sys
import threading
import time

import numpy as np
import sounddevice as sd
import soundfile as sf
import soxr
import torch
import yaml
from transformers.cache_utils import StaticCache

from model import Model

SR = 16000
FRAME = 320  # samples per codec token (50 tokens/s)
XFADE = 2 * FRAME  # crossfade between consecutive output segments
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    model = Model(config=config)
    # The released ckpt pickles a pathlib.PosixPath, which cannot be instantiated on Windows.
    if sys.platform == 'win32':
        pathlib.PosixPath = pathlib.WindowsPath
    ckpt = torch.load(config['ckpt_path'], map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    model.to(DEVICE)
    return model, config


def read_wav_16k_mono(path):
    wav, fs = sf.read(path, dtype='float32', always_2d=True)
    wav = wav[:, 0]
    if fs != SR:
        wav = soxr.resample(wav, fs, SR)
    return wav.astype(np.float32)


def prepare_enroll(wav, enroll_duration):
    # Same as the training dataloader: wrap-pad / cut to enroll_duration, peak-normalize.
    length = int(enroll_duration * SR)
    if wav.shape[-1] < length:
        wav = np.pad(wav, (0, length - wav.shape[-1]), mode='wrap')
    else:
        wav = wav[:length]
    return wav / (np.max(np.abs(wav)) + 1e-5) * 0.99


class TSE:
    """Greedy UniSE TSE over a sliding window, reusing the tokens of the previous window as history."""

    DECODE_CONTEXT = 64  # extra token frames decoded before the output: BiCodec's decoder is local, ~±40 frames

    def __init__(self, model, enroll, half=False, lookahead=0):
        self.model = model
        self.lookahead = lookahead  # token frames heard past the output; they are regenerated next step, not committed
        self.dtype = torch.float16 if half else torch.float32
        model.dnn.to(self.dtype)
        enroll = torch.from_numpy(enroll).float()[None].to(DEVICE)
        with torch.inference_mode():
            self.enroll_feats = model.extract_semantic_features(enroll).to(self.dtype)
            # Speaker tokens straight from the reference, as the training targets' were made, instead of
            # letting the model guess them from live audio (which may not even contain the target yet).
            tok = model.tokenizer
            self.global_ids = tok.model.get_global_tokens({'ref_wav': tok.get_ref_clip(enroll)}).reshape(1, -1).long()
        self.max_len = 0
        self.reset()

    def reset(self):
        self.semantic_ids = None

    @torch.inference_mode()
    def __call__(self, window, n_new, n_tail):
        """window: latest audio, a multiple of FRAME long; its last n_new samples are new since the last call.
        Returns the extracted audio for the n_tail samples ending `lookahead` frames before the window's end."""
        peak = np.max(np.abs(window))
        mix = torch.from_numpy(window / (peak + 1e-5) * 0.9).float()[None].to(DEVICE)
        mix_feats = self.model.extract_semantic_features(mix).to(self.dtype)
        total = window.shape[-1] // FRAME
        shift = n_new // FRAME
        # Tokens of the previous window, shifted by the new audio, are the history of this one.
        if self.semantic_ids is None or shift >= self.semantic_ids.size(1):
            history = self.global_ids.new_zeros((1, 0))
        else:
            history = self.semantic_ids[:, shift:]
        self.semantic_ids = self._generate(mix_feats, total, history)[:, :total - self.lookahead]
        frames = -(-n_tail // FRAME) + self.DECODE_CONTEXT
        est = self.model.tokenizer.detokenize(self.global_ids.unsqueeze(1), self.semantic_ids[:, -frames:]).reshape(-1)
        return est[-n_tail:].cpu().numpy().astype(np.float32)

    def _build(self, max_len):
        dnn = self.model.dnn
        self.cache = StaticCache(config=dnn.config, batch_size=1, max_cache_len=max_len, device=DEVICE, dtype=self.dtype)
        self.ids = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
        self.pos = torch.zeros(1, dtype=torch.long, device=DEVICE)

        off, size = dnn.semantic_offset, dnn.semantic_size

        def step():
            h = dnn.llm_forward(dnn.codec_embedding(self.ids), past_key_values=self.cache,
                                use_cache=True, cache_position=self.pos).last_hidden_state
            # argmax over the semantic vocab slice == argmax after masking everything else to -inf
            self.ids.copy_(dnn.output_head(h)[:, -1, off:off + size].argmax(-1, keepdim=True) + off)
            self.pos.add_(1)

        self.step = step
        self.max_len = max_len
        if DEVICE.type != 'cuda':
            return
        # Eager decoding launches ~500 tiny kernels per token and is launch-bound on Windows,
        # so the decode step is captured once and replayed as a single CUDA graph.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        self.step = graph.replay

    def _next(self):
        self.step()
        return self.ids.clone()

    def _prefill(self, embeds):
        self.model.dnn.llm_forward(embeds, past_key_values=self.cache, use_cache=True,
                                   cache_position=torch.arange(embeds.size(1), device=DEVICE))

    def _generate(self, mix_feats, total, history):
        # Same sequence as LLM_SFT.generate: [task, enroll_sos, enroll, mix_sos, mix, g_sos, global x32, s_sos, semantic...]
        dnn = self.model.dnn
        token = lambda v: torch.full((1, 1), v, dtype=torch.long, device=DEVICE)
        prefix = torch.cat([
            dnn.task_embedding(token(dnn.task_map['tse'])),
            dnn.enroll_sos_embedding(token(0)), dnn.adapter(self.enroll_feats),
            dnn.mix_sos_embedding(token(0)), dnn.adapter(mix_feats),
        ], dim=1)
        seq = torch.cat([token(dnn.global_sos_token_id), self.global_ids + dnn.global_offset,
                         token(dnn.semantic_sos_token_id), history + dnn.semantic_offset], dim=1)
        p = prefix.size(1)
        if self.max_len < p + seq.size(1) + total:
            self._build(p + seq.size(1) + total)
        self.cache.reset()
        self._prefill(torch.cat([prefix, dnn.codec_embedding(seq[:, :-1])], dim=1))
        self.pos.fill_(p + seq.size(1) - 1)
        self.ids.copy_(seq[:, -1:])
        new = [self._next() for _ in range(total - history.size(1))]
        return torch.cat([history] + [t - dnn.semantic_offset for t in new], dim=1)


class Player:
    """Pull-based playback buffer fed by the processing loop.

    Waits until `prebuffer` samples are queued before playing (again, after an underrun),
    so jitter in step times does not chop the audio into gaps."""

    def __init__(self, prebuffer):
        self.buf = np.zeros(0, dtype=np.float32)
        self.prebuffer = prebuffer
        self.playing = False
        self.lock = threading.Lock()

    def push(self, wav):
        with self.lock:
            self.buf = np.concatenate([self.buf, wav])

    def pending(self):
        with self.lock:
            return self.buf.shape[0]

    def callback(self, outdata, frames, time_info, status):
        with self.lock:
            if not self.playing and self.buf.shape[0] < self.prebuffer:
                outdata.fill(0.0)
                return
            self.playing = True
            n = min(frames, self.buf.shape[0])
            outdata[:n, 0] = self.buf[:n]
            outdata[n:, 0] = 0.0
            self.buf = self.buf[n:]
            if n < frames:
                self.playing = False


def record_enroll(seconds, device):
    print(f'\nRecording the TARGET speaker for {seconds:.0f} s. Only the target should talk. Starting in 2 s...')
    time.sleep(2)
    print('>>> Speak now')
    wav = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype='float32', device=device)
    sd.wait()
    print('<<< Done')
    return wav[:, 0]


def file_feeder(path, in_q, blocksize, stop):
    # Simulates a microphone by pushing a wav file in real time.
    wav = read_wav_16k_mono(path)
    for i in range(0, wav.shape[0], blocksize):
        if stop.is_set():
            break
        in_q.put(wav[i:i + blocksize].copy())
        time.sleep(blocksize / SR)
    in_q.put(None)  # end of input


def main(args):
    if args.list_devices:
        print(sd.query_devices())
        print('default (input, output):', sd.default.device)
        return 0

    torch.set_num_threads(args.threads)
    print(f'Using device: {DEVICE}')
    print('Loading models...')
    model, config = load_model(args.config)
    enroll_duration = config['enroll_duration']

    if args.enroll is not None:
        enroll_wav = read_wav_16k_mono(args.enroll)
    elif args.record_enroll is not None:
        enroll_wav = record_enroll(args.record_enroll, args.input_device)
        sf.write(args.save_enroll, enroll_wav, SR)
        print(f'Saved reference to {args.save_enroll} (reuse it with --enroll {args.save_enroll})')
    else:
        print('Need a reference of the target voice: pass --enroll <wav> or --record_enroll <seconds>.')
        return 1
    lookahead = int(args.lookahead * SR) // FRAME
    tse = TSE(model, prepare_enroll(enroll_wav, enroll_duration), half=DEVICE.type == 'cuda' and not args.fp32,
              lookahead=lookahead)

    window_len = int(args.window * SR) // FRAME * FRAME
    hop_len = max(int(args.hop * SR) // FRAME, 1) * FRAME
    print('Warming up...')
    tse(np.zeros(window_len, dtype=np.float32), window_len, hop_len + XFADE)
    tse(np.zeros(window_len, dtype=np.float32), hop_len, hop_len + XFADE)
    tse.reset()

    blocksize = SR // 10  # a multiple of FRAME, so new audio always ends on a token boundary
    in_q = queue.Queue()
    stop = threading.Event()
    player = Player(int(args.prebuffer * SR))
    played, captured = [], []

    def mic_callback(indata, frames, time_info, status):
        in_q.put(indata[:, 0].copy())

    streams = []
    if args.input_wav is not None:
        threading.Thread(target=file_feeder, args=(args.input_wav, in_q, blocksize, stop), daemon=True).start()
    else:
        streams.append(sd.InputStream(samplerate=SR, channels=1, dtype='float32', blocksize=blocksize,
                                      device=args.input_device, latency='low', callback=mic_callback))
    if not args.no_play:
        streams.append(sd.OutputStream(samplerate=SR, channels=1, dtype='float32', blocksize=blocksize,
                                       device=args.output_device, latency='low', callback=player.callback))
    for s in streams:
        s.start()

    print(f'\nLive TSE running: {args.window:.1f} s window, steps of >= {hop_len / SR:.2f} s. '
          f'The first output comes once {args.window:.0f} s of audio is in. Ctrl+C to stop.')
    if args.input_wav is None and not args.no_play:
        print('Tip: use headphones, otherwise the speaker output feeds back into the mic.')

    window = np.zeros(window_len, dtype=np.float32)
    filled = new = 0  # samples received in total / since the last step
    tail = None  # last XFADE samples of the previous output, held back for the crossfade
    ramp = np.linspace(0.0, 1.0, XFADE, dtype=np.float32)
    t_start = time.perf_counter()
    ended = False

    def emit(wav):
        wav = np.clip(wav * args.gain, -1.0, 1.0)
        if args.no_play:
            played.append(wav)
        else:
            player.push(wav)

    try:
        while True:
            blocks = []
            if not ended:
                try:
                    blocks.append(in_q.get(timeout=0.5))
                except queue.Empty:
                    continue
            while True:  # drain everything that already arrived
                try:
                    blocks.append(in_q.get_nowait())
                except queue.Empty:
                    break
            if any(b is None for b in blocks):  # `None in blocks` would compare arrays elementwise
                ended = True
                blocks = [b for b in blocks if b is not None]
            if blocks:
                block = np.concatenate(blocks)
                captured.append(block)
                window = np.concatenate([window, block])[-window_len:]
                filled += block.shape[0]
                new += block.shape[0]
            if ended and new % FRAME:  # the last block of a file can end mid-token
                pad = FRAME - new % FRAME
                window = np.concatenate([window, np.zeros(pad, dtype=np.float32)])[-window_len:]
                new += pad
            if not (ended and new) and (filled < window_len or new < hop_len):  # at the end, flush what is left
                if ended:
                    break
                continue

            # Fallen more than a window behind (or first step): regenerate, play only the latest hop.
            resync = tse.semantic_ids is None or new >= window_len
            n_out = min(new, hop_len) if resync else new
            t0 = time.perf_counter()
            seg = tse(window, new, n_out + XFADE)  # the newest n_out samples plus the overlap with the previous tail
            took = time.perf_counter() - t0
            new = 0

            rms = float(np.sqrt(np.mean(window[:window_len - lookahead * FRAME][-n_out:] ** 2)))  # input under this output
            if rms < args.silence_rms:
                seg = np.zeros_like(seg)
            head = seg[:XFADE] * ramp if tail is None else tail * ramp[::-1] + seg[:XFADE] * ramp
            emit(np.concatenate([head, seg[XFADE:n_out]]))
            tail = seg[n_out:]
            print(f'[{time.perf_counter() - t_start:6.1f}s] {n_out / SR:.2f} s of audio, step took {took:.2f} s'
                  f'{" (resync)" if resync else ""}{" (silent)" if rms < args.silence_rms else ""}')
            if ended and new == 0 and in_q.empty():
                emit(tail)
                break
        while not args.no_play and player.pending() > 0:  # let the tail play out
            time.sleep(0.1)
    except KeyboardInterrupt:
        print('\nStopping...')
    finally:
        stop.set()
        for s in streams:
            s.stop()
            s.close()

    if args.output_wav is not None:
        out = np.concatenate(played) if played else np.zeros(0, dtype=np.float32)
        sf.write(args.output_wav, out, SR)
        print(f'Wrote extracted audio to {args.output_wav}')
    if args.save_input is not None and captured:
        sf.write(args.save_input, np.concatenate(captured), SR)
        print(f'Wrote captured input to {args.save_input}')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser('live target speaker extraction')
    parser.add_argument('--config', type=str, default='./conf/config.yaml')
    parser.add_argument('--enroll', type=str, default=None, help='Reference wav of the target speaker.')
    parser.add_argument('--record_enroll', type=float, default=None, help='Record N seconds of the target speaker from the mic.')
    parser.add_argument('--save_enroll', type=str, default='enroll_recorded.wav', help='Where to save a recorded reference.')
    parser.add_argument('--window', type=float, default=5.0, help='Seconds of context the model sees (it was trained on 5 s).')
    parser.add_argument('--hop', type=float, default=0.5, help='Minimum seconds of new audio per step; larger hops are taken automatically when a step runs long.')
    parser.add_argument('--lookahead', type=float, default=0.25, help='Seconds of audio the model hears past each output before committing it (adds this much delay).')
    parser.add_argument('--prebuffer', type=float, default=0.25, help='Seconds of output queued before playback starts, to absorb step-time jitter.')
    parser.add_argument('--silence_rms', type=float, default=0.003, help='New audio quieter than this is output as silence.')
    parser.add_argument('--gain', type=float, default=1.0, help='Output gain.')
    parser.add_argument('--fp32', action='store_true', help='Run the language model in fp32 on GPU (slower; fp16 is the default there).')
    parser.add_argument('--threads', type=int, default=4, help='CPU threads for torch.')
    parser.add_argument('--input_device', type=int, default=1)  # Microphone (CDS.KT USB Audio)
    parser.add_argument('--output_device', type=int, default=7)  # Headphones (CDS.KT USB Audio)
    parser.add_argument('--list_devices', action='store_true')
    parser.add_argument('--input_wav', type=str, default=None, help='Feed a wav file in real time instead of the mic.')
    parser.add_argument('--output_wav', type=str, default=None, help='Also save the extracted audio here.')
    parser.add_argument('--save_input', type=str, default=None, help='Save the captured mic audio here.')
    parser.add_argument('--no_play', action='store_true', help='Do not open the speaker.')
    sys.exit(main(parser.parse_args()))
