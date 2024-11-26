import argparse
import os
from collections import defaultdict
from pathlib import Path

import librosa
import numpy
import numpy as np
import soundfile as sf
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from torch.nn import functional as F
from tqdm import tqdm
from transformers import pipeline


SPK_MODEL = {
    "speechbrain/spkrec-xvect-voxceleb": 512,
    "speechbrain/spkrec-ecapa-voxceleb": 192,
}


def f2embed(wav_file, classifier, size_embed):
    signal, fs = torchaudio.load(wav_file)
    if fs != 16000:
        signal = torchaudio.functional.resample(signal, fs, 16000)
    with torch.no_grad():
        embeddings = classifier.encode_batch(signal)
        embeddings = F.normalize(embeddings, dim=2)
        embeddings = embeddings.squeeze().cpu().numpy()
    assert embeddings.shape[0] == size_embed, embeddings.shape[0]
    return embeddings


class SpeechBrainSpeakerEmbeddingExtractor:
    def __init__(self, args):
        self.args = args
        device = "cuda" if torch.cuda.is_available() else "cpu"
        assert device == "cuda"
        self.classifier = EncoderClassifier.from_hparams(
            source=args.speaker_embed, run_opts={"device": device}, savedir=os.path.join("/tmp", args.speaker_embed)
        )
        self.size_embed = SPK_MODEL[args.speaker_embed]

    def __call__(self, wav_path):
        utt_emb = f2embed(wav_path, self.classifier, self.size_embed)
        numpy.save(wav_path.replace(".wav", ".npy"), utt_emb)


# args dictionary to attribute
class Dict2Attr:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--text_file",
        type=str,
        default="./data/synth_sentences.txt",
        help="text file path that containing reference wav path and text",
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, default="./examples/synth_data", help="output directory of synthesized speech"
    )
    parser.add_argument("-d", "--data_dir", type=str, default="./data", help="input data directory")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    speect5_args = {
        "arctic_root": args.output_dir,
        "output_root": args.output_dir,
        "speaker_embed": "speechbrain/spkrec-xvect-voxceleb",
        "splits": "p307",
    }
    arspeect5_args = Dict2Attr(**speect5_args)  # Dictionary to attribute
    speaker_embedding_extractor = SpeechBrainSpeakerEmbeddingExtractor(
        arspeect5_args
    )  # Load speech brain speaker embedding extractor
    synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts", device=device)  # Load SpeechT5 model

    # Load text file and make a dictionary with reference wav path as key
    # and text as value
    data = defaultdict(list)
    with open(args.text_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            w, t = line.strip().split("\t")  # w: reference wav path, t: text
            data[w].append(t.strip())

    ref_wav_list = sorted(list(data.keys()))

    for ref_speech_path in tqdm(ref_wav_list, desc="Synthesizing speech"):
        p = Path(ref_speech_path)
        spk = p.parts[0]
        os.makedirs(os.path.join(args.output_dir, spk), exist_ok=True)

        # We need to first extract speaker embedding to run SpeechT5-based
        # zero-shot TTS model.
        speaker_embedding_extractor(os.path.join(args.data_dir, ref_speech_path))
        speaker_embedding = (
            torch.tensor(np.load(os.path.join(args.data_dir, ref_speech_path.replace(".wav", ".npy"))))
            .unsqueeze(0)
            .cuda()
        )
        print("Extracting speaker embedding done.")

        for i, text in enumerate(data[ref_speech_path]):
            # Synthesize speech using SpeechT5 model
            speech = synthesiser(text, forward_params={"speaker_embeddings": speaker_embedding})
            # Write synthesized speech to a wav file
            sf.write(
                os.path.join(args.output_dir, spk, f"{spk}_speecht5_{i:03d}.wav"),
                librosa.util.normalize(speech["audio"]),
                samplerate=speech["sampling_rate"],
            )
