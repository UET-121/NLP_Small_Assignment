# Hệ Thống Dịch Máy Anh - Việt (IWSLT 2015): Transformer (PyTorch) vs Seq2Seq Bi-LSTM + Scaled Luong Attention (`tensorflow/nmt`)

Dự án tái hiện (**reimplementation**), huấn luyện và so sánh toàn diện 2 kiến trúc Dịch máy thần kinh (**Neural Machine Translation - NMT**) kinh điển trên bộ dữ liệu song ngữ **English - Vietnamese Parallel Corpus (IWSLT 2015)**:

1. **Case 1 — Mô hình Transformer (`demo_transformer.ipynb` / Vaswani et al., 2017)** bằng **PyTorch**:
   - **Kiến trúc**: 6 tầng Encoder & 6 tầng Decoder ($d_{\text{model}}=512$, $h=8$ Attention Heads, $d_{ff}=2048$, Pre-LayerNorm, Sinusoidal Positional Encoding).
   - **Huấn luyện**: Bộ điều chỉnh tốc độ học Noam `ScheduledOptim` ($\text{warmup}=4000$, $\beta=(0.9, 0.98)$) kết hợp hàm mất mát `LabelSmoothingLoss` ($\epsilon=0.1$).
   - **Giải mã**: Hỗ trợทั้ง Greedy Decoding theo lô (Batched Greedy) và Beam Search ($k=5$, length penalty $\alpha=0.7$).

2. **Case 2 — Mô hình Seq2Seq Bi-LSTM + Scaled Luong Attention (`https://github.com/tensorflow/nmt` / Luong et al., 2015, 2017)** bằng **TensorFlow / Keras (`tf.keras`)**:
   - **Kiến trúc**:
     - **Encoder 2 tầng LSTM**: Tầng 1 Bidirectional LSTM ($256 \times 2 = 512$) + Tầng 2 Unidirectional LSTM ($512$) kèm kết nối tắt (residual connection).
     - **Decoder 2 tầng LSTM**: 2 tầng Unidirectional LSTM ($512$) có kết nối tắt và cơ chế **Scaled Multiplicative Luong Attention**:
       $$\text{score}(h_t, \bar{h}_s) = \frac{h_t^\top W_a \bar{h}_s}{\sqrt{d}}, \qquad \alpha_{ts} = \text{softmax}_s(\text{score}(h_t, \bar{h}_s)), \qquad c_t = \sum_s \alpha_{ts}\bar{h}_s, \qquad \tilde{h}_t = \tanh(W_c[c_t; h_t])$$
   - **Huấn luyện**: `Adam` ($\text{lr}=10^{-3}$, gradient clipping `clipnorm=5.0`, giảm nửa tốc độ học sau epoch 6).
   - **Giải mã**: Hỗ trợ Batched Greedy Decoding và Batched-Step Beam Search ($k=5$).

---

## Cấu Trúc Thư Mục

```text
transformers/
├── data/                                   # Bộ dữ liệu song ngữ IWSLT 2015 En-Vi
│   ├── train.en / train.vi                 # Tập huấn luyện (133,166 cặp câu; 122,206 cặp <= 50 từ)
│   ├── tst2012.en / tst2012.vi             # Tập Validation / Dev (1,553 cặp câu)
│   └── tst2013.en / tst2013.vi             # Tập Test chuẩn quốc tế (1,268 cặp câu)
├── src/transformers/                       # Package Python chính
│   ├── __init__.py                         # Export toàn bộ API của dự án
│   ├── data_utils.py                       # Giải nén an toàn, Tokenizer EN/VI, Vocab, Field, BucketBatchSampler
│   ├── metrics.py                          # Tính điểm BLEU-1..4, Corpus BLEU (%), SacreBLEU
│   ├── transformer_pytorch.py              # Case 1: Mô hình Transformer bằng PyTorch
│   ├── nmt_tensorflow.py                   # Case 2: Mô hình Seq2Seq Bi-LSTM + Scaled Luong bằng TF/Keras
│   └── train_compare.py                    # Script huấn luyện, đánh giá trên tst2013 và vẽ biểu đồ
├── checkpoints/                            # Trọng số mô hình tốt nhất và bộ từ vựng
│   ├── vocab.json                          # Từ vựng EN (17,200 từ) và VI (7,666 từ)
│   ├── transformer_best.pt                 # Checkpoint Case 1 (PyTorch Transformer)
│   └── nmt_seq2seq_best.weights.h5         # Checkpoint Case 2 (TF/Keras Seq2Seq Luong)
├── results/                                # Kết quả đánh giá định lượng & biểu đồ so sánh
│   ├── comparison_results.json             # Toàn bộ chỉ số thực nghiệm của cả 2 mô hình
│   ├── sample_translations.json            # Các câu dịch đối chiếu trên tập tst2013
│   ├── training_curves.png                 # Đồ thị Loss, Perplexity, Dev BLEU qua 10 epochs
│   ├── metrics_comparison.png              # Biểu đồ cột so sánh BLEU-1..4 & SacreBLEU trên tst2013
│   ├── length_bucket_bleu.png              # Điểm BLEU theo độ dài câu nguồn trên tst2013
│   ├── transformer_attention_heatmap.png   # Heatmap Multi-Head Cross-Attention của Transformer
│   └── nmt_luong_attention_heatmap.png     # Heatmap Scaled Luong Attention của Seq2Seq NMT
├── demo_transformer.ipynb                  # Notebook Case 1 (tương thích Python 3.12 + PyTorch 2.x)
└── comparison_en_vi_nmt.ipynb              # Notebook tổng hợp so sánh Case 1 & Case 2 trên IWSLT 2015
```

---

## Kết Quả Thực Nghiệm Trên Tập Test `tst2013` (1,268 Câu)

| Mô hình & Chiến lược giải mã | Số tham số | Test Loss / PPL | BLEU-1 (%) | BLEU-2 (%) | BLEU-3 (%) | BLEU-4 / Corpus BLEU (%) | SacreBLEU |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Case 1: PyTorch Transformer (Greedy)** | 60,804,594 | 2.0987 / 8.16 | 55.42 | 41.51 | 31.60 | **24.30** | **24.30** |
| **Case 1: PyTorch Transformer (Beam $k=5$)** | 60,804,594 | 2.0987 / 8.16 | **57.33** | **43.37** | **33.36** | **25.96** | **25.96** |
| **Case 2: TF/Keras NMT Bi-LSTM+Luong (Greedy)** | 25,315,328 | 2.3480 / 10.47 | 53.14 | 38.98 | 29.15 | **22.07** | **22.07** |
| **Case 2: TF/Keras NMT Bi-LSTM+Luong (Beam $k=5$)** | 25,315,328 | 2.3480 / 10.47 | 53.24 | 40.11 | 30.65 | **23.61** | **23.61** |

### Điểm BLEU (%) Theo Độ Dài Câu Nguồn trên `tst2013` (Beam Search $k=5$)

| Độ dài câu tiếng Anh (số từ) | Số câu (`tst2013`) | Case 1: Transformer BLEU (%) | Case 2: NMT Bi-LSTM + Luong BLEU (%) | Chênh lệch (Transformer - NMT) |
|---|---:|---:|---:|---:|
| `< 10` | 231 | **31.15%** | 29.55% | **+1.60%** |
| `10 - 19` | 454 | **29.90%** | 26.11% | **+3.79%** |
| `20 - 29` | 295 | **26.97%** | 24.63% | **+2.34%** |
| `30 - 39` | 156 | **26.19%** | 23.96% | **+2.23%** |
| `>= 40` | 132 | **19.08%** | 18.52% | **+0.56%** |

---

## Hướng Dẫn Chạy

1. **Huấn luyện và đánh giá lại từ đầu qua CLI**:
   ```bash
   .venv/bin/python -m transformers.train_compare --epochs-transformer 10 --epochs-nmt 10 --batch-size 128
   ```
2. **Thực thi các Jupyter Notebooks**:
   ```bash
   .venv/bin/jupyter nbconvert --to notebook --execute --inplace demo_transformer.ipynb
   .venv/bin/jupyter nbconvert --to notebook --execute --inplace comparison_en_vi_nmt.ipynb
   ```
