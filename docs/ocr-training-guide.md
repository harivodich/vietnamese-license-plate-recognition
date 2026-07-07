# OCR training guide

## Mục tiêu

OCR baseline pretrained đã cho thấy PaddleOCR đọc tốt hơn trên biển một dòng rộng, nhưng gần như thất bại với biển hai dòng compact. Vì vậy bước này chuẩn bị một baseline OCR có thể train lại bằng dữ liệu của project:

```text
OCR crop
  -> nếu biển compact: tách thành dòng trên + dòng dưới
  -> resize/pad từng dòng về ảnh xám 32 x 160
  -> CRNN đọc chuỗi ký tự
  -> CTC loss học alignment ảnh-ký tự mà không cần gán vị trí từng chữ
```

Bước này chưa phải train full. Nó tạo code, config, dữ liệu line-level, checkpoint/resume flow và smoke-test để khi train qua đêm không bị thiếu hạ tầng.

## Vì sao dùng CRNN + CTC cho train local

PaddleOCR vẫn được giữ làm pretrained baseline. Nhưng fine-tune PaddleOCR trên laptop hiện tại không phải lựa chọn local tốt nhất vì GPU Quadro M2200 quá cũ cho bản PaddlePaddle GPU hiện hành. CRNN + CTC dùng PyTorch, nhẹ hơn, và đã khớp với stack detection đang chạy được trên máy này.

Trade-off:

- Ưu điểm: nhẹ, dễ đọc code, checkpoint/resume rõ, phù hợp học cơ chế OCR sequence.
- Nhược điểm: không mạnh bằng các OCR hiện đại có attention/transformer; không tự xử lý layout phức tạp nếu crop hai dòng chưa được tách tốt.

## Dữ liệu train OCR được tạo như thế nào

Lệnh chuẩn bị dữ liệu:

```powershell
python scripts/prepare_ocr_training.py --config configs/ocr-crnn.yaml
```

File cấu hình chính: `configs/ocr-crnn.yaml`.

Input là manifest OCR đã audit: `data/processed/ocr_manifest.jsonl`.

Output nằm trong `data/processed/ocr_training/` và không push lên Git:

- `images/train/`: ảnh dòng dùng để train;
- `images/validation/`: ảnh dòng dùng để validation;
- `train.txt`: mỗi dòng là `relative_image_path<TAB>label`;
- `validation.txt`: cùng format với train;
- `charset.txt`: danh sách ký tự model được phép dự đoán;
- `summary.json`: số sample train/validation và số ký tự.

Số mẫu hiện tại sau khi materialize:

| Split | Line samples |
| --- | ---: |
| Train | 7,595 |
| Validation | 1,249 |

Baseline train đầu tiên dùng `include_compact: false`, nên chỉ giữ biển một dòng rộng. Biển hai dòng compact tạm thời bị loại khỏi training để tránh nhiễu từ bước split tự động. Sau khi one-line baseline học ổn, compact sẽ được đưa lại như một thí nghiệm riêng.

## CRNN là gì

CRNN gồm ba phần:

```text
image line 32x160
  -> CNN trích đặc trưng thị giác
  -> BiLSTM đọc đặc trưng theo chiều ngang trái-phải
  -> Linear dự đoán xác suất ký tự tại từng timestep
```

CNN trả lời: vùng này nhìn giống nét chữ nào?

LSTM trả lời: chuỗi nét theo chiều ngang tạo thành thứ tự ký tự nào?

Linear trả lời: ở timestep này xác suất là `0`, `1`, `A`, `B`, `Đ`, hoặc blank là bao nhiêu?

## CTC loss là gì

CTC dùng khi ta biết label cuối cùng, ví dụ `51G`, nhưng không biết chính xác chữ `5`, `1`, `G` nằm ở timestep nào trong ảnh.

Model có thể dự đoán nhiều timestep hơn số ký tự:

```text
-- 5 5 blank 1 blank G G --
```

CTC collapse chuỗi này thành:

```text
51G
```

Vì vậy ta không cần annotate từng ký tự bằng bbox riêng. Đây là lý do CTC phù hợp với OCR biển số khi dataset chỉ có ảnh crop và text label.

## Các lệnh cần chạy

Kiểm tra config, charset, label, ảnh, CUDA và độ dài CTC nhưng không train:

```powershell
python scripts/train_ocr.py --config configs/ocr-crnn.yaml --check-only
```

Chạy smoke-test một batch train + một lượt validation nhỏ:

```powershell
python scripts/train_ocr.py --config configs/ocr-crnn.yaml --smoke-test
```

Chạy tiny-overfit trước khi train full:

```powershell
python scripts/train_ocr.py --config configs/ocr-crnn.yaml --tiny-overfit
```

Tiny-overfit bắt model học thuộc vài ảnh OCR thật. Nếu bước này fail, không nên train full vì model/loss/preprocessing đang có lỗi cơ bản.

Train full:

```powershell
python scripts/train_ocr.py --config configs/ocr-crnn.yaml
```

Resume nếu máy tắt hoặc bị dừng:

```powershell
python scripts/train_ocr.py --config configs/ocr-crnn.yaml --resume artifacts/ocr/crnn-ctc-wide-baseline/last.pt
```

Trong CMD nên viết một dòng. Trong PowerShell có thể xuống dòng bằng dấu backtick.

## Train xong lưu những gì

Trainer lưu vào `artifacts/ocr/crnn-ctc-wide-baseline/`:

- `last.pt`: checkpoint mới nhất, dùng để resume;
- `best.pt`: checkpoint tốt nhất theo validation exact match;
- `epoch_005.pt`, `epoch_010.pt`, ...: checkpoint định kỳ theo `save_period`;
- `history.csv`: log từng epoch gồm loss và metric;
- `training_curves.png`: đồ thị train loss, validation loss, exact match, CER.

Thư mục `artifacts/` được gitignore, nên checkpoint nặng không bị push lên GitHub.

## Đọc chỉ số trong lúc train

Các metric chính:

- `train_loss`: loss trên train set. Nên giảm dần, nhưng không phải metric cuối cùng.
- `validation_loss`: loss trên validation. Nếu train loss giảm mà validation loss tăng, model có thể overfit.
- `exact_match`: tỷ lệ đọc đúng toàn bộ dòng. Đây là metric quan trọng nhất cho OCR biển số.
- `CER`: character error rate. Thấp hơn là tốt hơn.
- `character_accuracy`: tỷ lệ ký tự đúng. Cao hơn là tốt hơn.

Ví dụ đọc kết quả:

```text
exact_match tăng, CER giảm
```

nghĩa là model đang học đúng hướng.

```text
train_loss giảm mạnh, exact_match validation không tăng
```

nghĩa là model có thể đang học thuộc train hoặc preprocessing/tách dòng còn lỗi.

## Các tham số quan trọng trong config

`image_height: 32`: chiều cao phổ biến cho OCR line. Nhỏ đủ nhanh, vẫn giữ nét chữ.

`image_width: 160`: canvas ngang cố định sau resize/pad. Nếu plate dài hơn nhiều, tăng width sẽ giúp nhưng tốn VRAM hơn.

`hidden_size: 256`: kích thước LSTM. OCR line nhỏ nhưng chuỗi có độ dài biến thiên, nên baseline dùng 256 để tránh underfit sớm.

`batch_size: 64`: số ảnh dòng mỗi optimizer step. Nếu lỗi CUDA out of memory, giảm xuống `32` hoặc `16`.

`blank_bias: -2.0`: giảm bias ban đầu của CTC blank class. CTC dễ collapse về blank hoặc chuỗi ngắn phổ biến; bias âm buộc model thử ký tự thật sớm hơn.

`learning_rate: 0.001`: tốc độ cập nhật trọng số. Nếu loss dao động mạnh, giảm xuống `0.0005`.

`min_epochs: 30`: không cho early stopping dừng trước 30 epoch, vì OCR từ scratch thường chưa có exact match ở giai đoạn đầu.

`patience: 15`: dừng sớm nếu checkpoint không còn cải thiện trong 15 epoch sau mốc tối thiểu. Cải thiện được xét theo exact match, rồi CER, rồi validation loss.

`gradient_clip_norm: 5.0`: chặn gradient quá lớn để training LSTM ổn định hơn.

`save_period: 5`: lưu checkpoint định kỳ mỗi 5 epoch.

## Khi áp dụng sang bài OCR khác cần đổi gì

Bắt buộc đổi:

- `manifest` hoặc label file nguồn;
- rule normalize text;
- `charset.txt` nếu bộ ký tự khác;
- cách xử lý layout nếu ảnh không phải biển số một dòng/hai dòng.

Có thể cần đổi:

- `image_width` nếu chuỗi dài hơn;
- augmentation nếu ảnh mờ, nghiêng, tối nhiều hơn;
- model lớn hơn nếu dataset nhiều và đa dạng hơn;
- metric chính nếu bài toán cho phép sai một vài ký tự.

Không nên đổi:

- test split sau khi đã bắt đầu so sánh model;
- validation rule giữa các thí nghiệm nếu muốn ablation công bằng.

## Checklist trước khi train qua đêm

```powershell
python scripts/prepare_ocr_training.py --config configs/ocr-crnn.yaml
python scripts/train_ocr.py --config configs/ocr-crnn.yaml --check-only
python scripts/train_ocr.py --config configs/ocr-crnn.yaml --smoke-test
python scripts/train_ocr.py --config configs/ocr-crnn.yaml --tiny-overfit
```

Nếu bốn lệnh trên pass, có thể bắt đầu train full. Nếu máy tắt, dùng lệnh `--resume` với `last.pt`.
