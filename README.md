# Image Captioning on MS COCO (CNN-RNN)

End-to-end image-captioning pipeline. A frozen **InceptionV3** encoder feeds a single-layer **LSTM** decoder, trained jointly with an **auxiliary multi-label classification head** over the 80 MS COCO object categories. The whole pipeline runs in a single Google Colab notebook on a free T4 GPU.

> Architecture is the classic *Show and Tell* (Vinyals et al., 2015): image feature → projected embedding → initial `(h, c)` of the LSTM → token-by-token softmax decoder.

---

## Team members

Yousef Magdy  
Jana Raed   
Kareem Elfeel

---

## Results

Evaluated on **1,000 held-out test images** carved from `train2014` before the 50% subsample (see "Data splits" below):

| Metric  | Score  |
|---------|--------|
| BLEU-1  | ≈ 0.67 |
| BLEU-2  | ≈ 0.50 |
| BLEU-3  | ≈ 0.35 |
| BLEU-4  | ≈ 0.25 |

For reference, the original *Show and Tell* paper reports BLEU-4 ≈ 0.277 with a much larger training run, fine-tuned CNN, and beam size 20.

Training metrics at the best epoch: val caption loss ≈ 2.27, val caption acc ≈ 53%, val aux AUC ≈ 0.95.

---

## Pipeline overview

1. **Download** MS COCO 2014 from Kaggle (`hariwh0/ms-coco-dataset`).
2. **Caption preprocessing:**  lowercase, strip punctuation, wrap with `<start>` / `<end>`, tokenize, pad to 34 tokens, vocab capped at 10,000 (full unique vocab is ~17K).
3. **Image preprocessing:**  resize 299×299, `inception_v3.preprocess_input`, run through frozen InceptionV3 to get 2048-d features, **cache to `.npy` once** so each epoch only touches disk.
4. **Two-output Functional Keras model:**
   - **Caption head:** `Embedding(10000, 256) → LSTM(512) → TimeDistributed(Dense(softmax))`
   - **Aux head:** `Dense(256, ReLU) → Dense(80, sigmoid)` for multi-label COCO categories
   - The image features project to 256-d, then to two `tanh` Dense layers used as the LSTM's initial `(h, c)` state.
5. **Loss:** `caption: 1.0 × sparse_categorical_crossentropy + aux: 0.2 × binary_crossentropy`.
6. **Training:** Adam(1e-3) for 10 epochs with `ReduceLROnPlateau`, `EarlyStopping(restore_best_weights=True)`, and `ModelCheckpoint`.
7. **Inference:** greedy + beam-3 caption generation.
8. **Eval:** corpus BLEU-1..4 (smoothing method 1) on 1,000 test images vs. five human references each.

---

## Data splits

The Kaggle mirror `hariwh0/ms-coco-dataset` contains only the 2014 **training** images, the 2014 val/test image folders are not in the archive (only `captions_val2014.json` is shipped, and its referenced images are absent). The notebook auto-detects this and falls back to carving val + test out of `train2014`:

| Split | Source | Count |
|-------|--------|------:|
| **Test**  | 5% of `train2014` (held out before any subsampling)  | 4,139 |
| **Val**   | 5% of `train2014` (held out before any subsampling)  | 4,139 |
| **Train** | remaining 90%, then 50% random subsample             | 37,252 |

All splits are carved with a fixed random seed *before* the 50% subsample, so training never sees val or test images. BLEU is reported on 1,000 images drawn from the held-out test split.

---

## Repo layout

```
.
├── AML_Project2_ImageCaptioning.ipynb   # the deliverable -> runs end-to-end on Colab
├── README.md
└── .gitignore
```

The notebook is the single artifact. Trained `caption_final.keras` + `tokenizer.pkl` are saved to `/content/models/` and copied to Google Drive at the end of the run; they are not committed to the repo.

---

## How to run

1. Open `AML_Project2_ImageCaptioning.ipynb` in **Google Colab**.
2. Switch runtime to **GPU (T4 is enough)**.
3. **Runtime → Run all**.
4. When prompted, upload your `kaggle.json` (Kaggle → Settings → Create API Token).
5. Dataset (~13 GB) downloads in ~15 min, feature extraction ~7 min, training ~5 min/epoch × 10 epochs, BLEU eval ~5 min.

Total wall-clock: roughly **75–90 min** on a free T4.

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
| `VOCAB_SIZE` | 10,000 | covers ~98% of token occurrences (full vocab ~17K) |
| `EMBED_DIM` | 256 | shared image / word embedding dim |
| `LSTM_UNITS` | 512 | standard Show-and-Tell setting |
| `BATCH_SIZE` | 64 | fits T4 VRAM at this model size |
| `EPOCHS` | 10 | val loss still slowly improving at 10 - 15 would help |
| `SUBSET_FRACTION` | 0.50 | spec minimum |
| `loss_weights` | caption 1.0 / aux 0.2 | aux is supervision aid, not the main objective |
| `optimizer` | Adam(1e-3) | with `ReduceLROnPlateau(factor=0.5, patience=1)` |
| `dropout` | 0.3 | applied on image projection, LSTM output, and aux MLP |

---

## Architecture choices

The assignment allows the use of *any* CNN/RNN, with the caveat that architectures *not* discussed in the course must be described carefully. We deliberately stayed within course-covered material:

- **InceptionV3:** a standard ImageNet-pretrained CNN already discussed in lectures, used as a frozen 2048-d feature extractor.
- **Single-layer LSTM:** the canonical RNN variant covered in class, used as the language decoder with image-conditioned `(h, c)` initial state.

Because both are course-standard, the spec's "describe any non-course architecture" clause does not apply here. A clear architectural walk-through is included in the notebook for completeness.

---

## Visualizations included in the notebook

- Interactive Plotly **training curves** (loss, caption accuracy, aux AUC) across all 10 epochs.
- **BLEU-1..4 bar chart** with score labels.
- **Caption-length histogram** comparing ground-truth vs. predicted lengths (sanity check that the model isn't collapsing into stock-length sentences).
- **Top-15 COCO categories** chart comparing aux-head predictions to ground-truth label counts.
- **Qualitative gallery** of 8 test images with ground truth, greedy, and beam-3 captions.

---

## Stack

`tensorflow 2.19`, `keras 3`, `pycocotools`, `nltk` (BLEU), `pandas`, `plotly`, `matplotlib`, `tqdm`.

---

## License

Educational / coursework, not for commercial use. MS COCO data is governed by its own license.
