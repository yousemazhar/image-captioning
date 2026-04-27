# Image Captioning on MS COCO (CNN-RNN)

End-to-end image-captioning pipeline built for **Advanced Machine Learning, Spring 2026 (GIU)** — Project 2.
A frozen **InceptionV3** encoder feeds a single-layer **LSTM** decoder, trained jointly with an **auxiliary multi-label classification head** over the 80 MS COCO object categories. The whole pipeline runs in a single Google Colab notebook on a free T4 GPU.

> Architecture is the classic *Show and Tell* (Vinyals et al., 2015) recipe — image feature → projected embedding → initial `(h, c)` of the LSTM → token-by-token softmax decoder.

---

## Results

Evaluated on **1,000 held-out test images** (5% test split carved from `train2014` — see "Data splits" below):

| Metric  | Score  |
|---------|--------|
| BLEU-1  | 0.679  |
| BLEU-2  | 0.498  |
| BLEU-3  | 0.352  |
| BLEU-4  | 0.250  |

For reference, the original *Show and Tell* paper reports BLEU-4 ≈ 0.277 with a much larger training run, fine-tuned CNN, and beam size 20.

Training metrics at epoch 10 (best): val caption loss = 2.27, val caption acc = 53.1%, val aux AUC = 0.951.

---

## Pipeline overview

1. **Download** MS COCO 2014 from Kaggle (`hariwh0/ms-coco-dataset`).
2. **Caption preprocessing** — lowercase, strip punctuation, wrap with `<start>` / `<end>`, tokenize, pad to 34 tokens, vocab capped at 10,000.
3. **Image preprocessing** — resize 299×299, `inception_v3.preprocess_input`, run through frozen InceptionV3 to get 2048-d features, **cache to `.npy` once**.
4. **Two-output Functional Keras model** —
   - Caption head: `Embedding → LSTM(512) → TimeDistributed(Dense(softmax))`
   - Aux head: `Dense(256) → Dense(80, sigmoid)` for multi-label COCO categories
   - Image features project to 256-d, then to two `tanh` Dense layers used as the LSTM's initial `(h, c)` state.
5. **Loss** — `caption: 1.0 × sparse_categorical_crossentropy + aux: 0.2 × binary_crossentropy`.
6. **Inference** — greedy + beam-3 caption generation.
7. **Eval** — BLEU-1..4 (corpus-level, smoothing method 1) on 1,000 test images.

---

## Data splits

The Kaggle mirror `hariwh0/ms-coco-dataset` contains only the 2014 **training** images; the 2014 val/test image folders are not in the archive (only `captions_val2014.json` is shipped, and its referenced images are absent). The notebook auto-detects this and falls back to:

| Split | % of train2014 | Count |
|-------|-----------------|-------|
| Test  | 5%             | ~4,139 |
| Val   | 5%             | ~4,139 |
| Train | 90%, then 50% subsampled | ~39,322 |

All splits are carved with a fixed random seed *before* the 50% subsample, so training never sees val or test images.

---

## Repo layout

```
.
├── AML_Project2_ImageCaptioning.ipynb   # the deliverable — runs end-to-end on Colab
└── README.md
```

The notebook is the single artifact. Trained `.keras` model + `tokenizer.pkl` are saved to `/content/models/` and copied to Google Drive at the end of the run; they are not committed to the repo.

---

## How to run

1. Open `AML_Project2_ImageCaptioning.ipynb` in **Google Colab**.
2. Switch runtime to **GPU (T4)**.
3. **Runtime → Run all**.
4. When prompted, upload your `kaggle.json` (Kaggle → Settings → Create API Token).
5. The dataset (~13 GB) downloads in ~15 min, feature extraction takes ~5 min, training is ~5 min/epoch × 10 epochs.

Total wall-clock: ~90 min on a free T4.

### Loading the trained model later

```python
import pickle, tensorflow as tf
model = tf.keras.models.load_model('caption_final.keras')
art = pickle.load(open('tokenizer.pkl', 'rb'))
# art['word_index'], art['index_word'], art['max_len'], art['idx_to_cat'], ...
```

---

## Hyperparameters

| Param | Value | Rationale |
|-------|-------|-----------|
| `IMG_SIZE` | 299 | InceptionV3 native input |
| `MAX_LEN` | 34 | covers 99th-pct caption length + `<start>`/`<end>` |
| `VOCAB_SIZE` | 10,000 | covers ~98% of token occurrences |
| `EMBED_DIM` | 256 | shared image / word embedding dim |
| `LSTM_UNITS` | 512 | standard Show-and-Tell setting |
| `BATCH_SIZE` | 64 | fits T4 VRAM at this model size |
| `EPOCHS` | 10 | val loss still slowly improving — could push to 15 |
| `SUBSET_FRACTION` | 0.50 | spec minimum |
| `loss_weights` | caption 1.0 / aux 0.2 | aux is supervision aid, not the objective |

---

## Stack

`tensorflow 2.19`, `keras 3`, `pycocotools`, `nltk` (for BLEU), `pandas`, `plotly`, `matplotlib`, `tqdm`.

---

## License

Educational / coursework — not for commercial use. MS COCO data is governed by its own license.
