## How to run an ASR experiment with LibriSpeech
To train a full speech recognition system the pipeline is the following:
1. **Train a tokenizer.** The tokenizer takes in input the training transcripts and determines the subword units that will be used for both acoustic and language model training. **Training a tokenizer before the language and acoustic model is necessary**. Indeed, both of them will reuse this tokenizer to map the output tokens.
2. **Train a Language Model (LM).** The language model takes in input long texts from available books. We have recipes with RNN and transformer-based LMs. In both cases, the LM is used during beam search to assign different weights to different hypotheses generated by the acoustic model.
3. **Train an acoustic model (AM).** The acoustic model maps the input speech into a set of sub-words units. The current repository contains recipes for seq2seq (ctc+attention), transducers, and transformer-based systems. Since training an LM can take several days, by default the recipes downloads a pre-trained LM.

**The results obtained with the different models can be found in the corresponding sub-directories!**

**Note:** *This folder also contains a Grapheme-to-phoneme  (G2P) system that can be used to convert a sequence of characters into the corresponding sequence of phonemes.*

## How to simply use pretrained models to transcribe my audio file?

SpeechBrain provides a simple interface to transcribe audio files with pretrained models. All the necessary information can be found on the different HuggingFace repositories corresponding to our different models for LibriSpeech:
- [seq2seq (ctc+attention) + RNNLM](https://huggingface.co/speechbrain/asr-crdnn-rnnlm-librispeech)
- [seq2seq (ctc+attention) + TransformerLM](https://huggingface.co/speechbrain/asr-crdnn-transformerlm-librispeech)
- [Transformer + ctc + TransformerLM](https://huggingface.co/speechbrain/asr-transformer-transformerlm-librispeech)



# **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/


# **Citing SpeechBrain**
Please, cite SpeechBrain if you use it for your research or business.

```bibtex
@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
  year={2021},
  eprint={2106.04624},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  note={arXiv:2106.04624}
}
```