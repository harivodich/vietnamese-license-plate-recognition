# Detection training guide

## Mục tiêu của baseline

Baseline trả lời một câu hỏi có kiểm soát:

> Với dữ liệu và split hiện tại, một detector YOLO pretrained cơ bản đạt chất lượng nào?

Baseline chưa nhằm tạo model tốt nhất. Nó tạo mốc đo đầu tiên để mọi cải tiến sau này phải chứng
minh tốt hơn trên cùng validation/test set.

## Luồng chương trình

```text
configs/detection-baseline.yaml
        |
        v
load_detection_experiment()
        |
        v
validate_detection_experiment()
        |
        v
build_training_arguments()
        |
        v
YOLO(pretrained weights).train(...)
        |
        +--> train batches: forward -> loss -> backward -> optimizer step
        |
        +--> validation: precision, recall, mAP
        |
        +--> best.pt, last.pt, results.csv, plots
```

Test set không tham gia vòng lặp trên. Nó chỉ được dùng sau khi checkpoint đã được chọn bằng
validation.

## Ba lệnh cần biết

Kiểm tra dữ liệu và config mà không train:

```powershell
python scripts/train_detection.py --config configs/detection-baseline.yaml --check-only
```

Bắt đầu training mới:

```powershell
python scripts/train_detection.py --config configs/detection-baseline.yaml
```

Tiếp tục nếu máy tắt hoặc training bị ngắt:

```powershell
python scripts/train_detection.py `
  --config configs/detection-baseline.yaml `
  --resume artifacts/detection/yolo11n-baseline/weights/last.pt
```

Trong CMD, viết lệnh trên một dòng hoặc thay backtick bằng `^`.

## Script entry point

`scripts/train_detection.py` chỉ có nhiệm vụ gọi module chính:

```python
from vlpr.training.detection import main

if __name__ == "__main__":
    raise SystemExit(main())
```

- `import main`: lấy hàm CLI từ package, tránh đặt business logic trong `scripts/`.
- `__name__ == "__main__"`: chỉ chạy khi file được gọi trực tiếp.
- `SystemExit`: chuyển mã `0` hoặc `1` về terminal và CI.

## Các schema cấu hình

### `DetectionTrainSettings`

Class này kiểm tra các tham số tối ưu:

- `epochs`: số lần model đi qua toàn bộ train set.
- `patience`: dừng sớm nếu validation không cải thiện trong số epoch này.
- `imgsz`: kích thước canvas đưa vào model. `640` là cân bằng phổ biến giữa chi tiết và tốc độ.
- `batch`: số ảnh dùng cho một optimizer step. Batch lớn ổn định hơn nhưng tốn VRAM hơn.
- `workers`: số process chuẩn bị dữ liệu song song.
- `device`: GPU được chọn. `0` nghĩa là CUDA GPU đầu tiên.
- `seed`: khởi tạo các phép ngẫu nhiên có thể tái lập.
- `deterministic`: ưu tiên phép toán xác định; có thể chậm hơn một chút.
- `amp`: dùng mixed precision để giảm VRAM và tăng tốc trên GPU.
- `cache`: có cache toàn bộ ảnh trong RAM/đĩa hay không.
- `val`: chạy validation trong quá trình train.
- `save`: lưu checkpoint.
- `save_period`: lưu checkpoint định kỳ; `5` nghĩa là mỗi 5 epoch.
- `plots`: xuất curve và visualization của Ultralytics.
- `optimizer`: quy tắc cập nhật trọng số. Baseline dùng AdamW.
- `lr0`: learning rate ban đầu.
- `lrf`: learning rate cuối bằng `lr0 * lrf`.
- `weight_decay`: regularization hạn chế trọng số tăng quá lớn.
- `warmup_epochs`: tăng learning rate từ từ ở đầu training.
- `cos_lr`: giảm learning rate theo cosine schedule.

Pydantic dùng `extra="forbid"`, nên gõ sai `epoch` thay vì `epochs` sẽ bị từ chối ngay thay vì bị
bỏ qua âm thầm.

### `DetectionAugmentationSettings`

Augmentation tạo biến thể từ ảnh train:

- `hsv_h/s/v`: đổi màu, độ bão hòa và độ sáng.
- `degrees`: xoay nhẹ.
- `translate`: dịch chuyển ảnh.
- `scale`: zoom in/out.
- `shear`: biến dạng xiên.
- `perspective`: biến đổi phối cảnh nhẹ.
- `flipud`: lật dọc; đặt `0` vì xe lộn ngược không thực tế.
- `fliplr`: lật ngang.
- `mosaic`: ghép bốn ảnh để tăng đa dạng scale/context.
- `mixup`: trộn hai ảnh và target.
- `close_mosaic`: tắt mosaic ở các epoch cuối để model hội tụ trên ảnh tự nhiên.

Augmentation chỉ áp dụng cho train. Validation và test phải phản ánh dữ liệu thật.

## Giải thích từng hàm

### `load_detection_experiment(path)`

1. Mở YAML bằng UTF-8.
2. `yaml.safe_load()` chuyển YAML thành Python dictionary.
3. Kiểm tra root phải là mapping.
4. `DetectionExperimentConfig.model_validate()` chuyển dictionary thành object có kiểu.

Hàm này chỉ kiểm tra cấu hình. Nó chưa mở ảnh và chưa tạo model.

### `validate_detection_experiment(path)`

Hàm này chạy trước khi dùng GPU:

1. Gọi `load_detection_experiment()`.
2. Suy ra repository root từ thư mục `configs`.
3. Resolve đường dẫn và chặn đường dẫn thoát khỏi repository.
4. Đọc YOLO dataset YAML.
5. Kiểm tra đủ `train`, `val`, `test`.
6. Đọc từng image list.
7. Bắt buộc ảnh nằm trong `data/processed/.../images`, không được trỏ vào raw.
8. Kiểm tra ảnh và label tồn tại.
9. Từ chối label rỗng.
10. Từ chối ảnh lặp trong một split hoặc xuất hiện ở nhiều split.

Preflight rẻ hơn nhiều so với phát hiện lỗi sau vài giờ training.

### `build_training_arguments(config_path, config)`

```python
arguments = {
    **config.train.model_dump(),
    **config.augmentation.model_dump(),
    "data": ...,
    "project": ...,
    "name": config.name,
}
```

- Hai toán tử `**` trải các field Pydantic thành keyword arguments.
- `data` được đổi thành đường dẫn tuyệt đối tới dataset YAML.
- `project` là thư mục chứa kết quả.
- `name` là tên experiment.

Hàm riêng này giúp unit test xác nhận chính xác tham số nào sẽ được gửi sang Ultralytics mà không
cần chạy GPU.

### `train_detection(config_path)`

```python
config = validate_detection_experiment(config_path)
model = YOLO(config.model)
model.train(**build_training_arguments(config_path, config))
```

1. Preflight dữ liệu.
2. `YOLO("yolo11n.pt")` tải/nạp pretrained weights.
3. Detection head được điều chỉnh từ class ImageNet/COCO sang một class `license_plate`.
4. `model.train()` chạy vòng lặp train và validation.

Đây là transfer learning. Model không học cạnh, texture và hình khối hoàn toàn từ đầu.

### `resume_detection(config_path, checkpoint_path)`

Resume dùng `last.pt`, không dùng `best.pt`:

- `last.pt` chứa trạng thái gần nhất của model, optimizer, scheduler và epoch.
- `best.pt` là checkpoint có validation fitness tốt nhất, dùng để đánh giá/inference.

```python
model = YOLO(str(checkpoint))
model.train(resume=True)
```

`resume=True` tiếp tục training state cũ. Không tự ý thay learning rate hoặc augmentation giữa một
run đang resume.

### `main(argv)`

`main()` phân ba chế độ:

- `--check-only`: chỉ preflight.
- `--resume LAST_PT`: tiếp tục run bị ngắt.
- Không có flag: bắt đầu run mới.

Các lỗi cấu hình dự kiến được log và trả exit code `1`. Thành công trả `0`.

## Một optimizer step diễn ra thế nào?

Với mỗi batch:

1. Data loader đọc ảnh và YOLO labels.
2. Ảnh được resize/letterbox và augmentation.
3. Model chạy **forward pass** để tạo prediction.
4. Prediction được so với ground truth để tính loss.
5. **Backward pass** tính gradient cho từng trọng số.
6. AdamW dùng gradient để cập nhật trọng số.
7. Gradient được xóa trước batch tiếp theo.

Một epoch hoàn thành khi model đã đi qua toàn bộ train set một lần.

## Cách đọc output

Kết quả nằm dưới:

```text
artifacts/detection/yolo11n-baseline/
├── args.yaml
├── results.csv
├── results.png
├── confusion_matrix.png
├── PR_curve.png
├── F1_curve.png
└── weights/
    ├── best.pt
    └── last.pt
```

### Training loss

- `box_loss`: sai số vị trí/kích thước bbox.
- `cls_loss`: sai số class. Dù chỉ có một class, model vẫn phải phân biệt plate và background.
- `dfl_loss`: Distribution Focal Loss giúp định vị cạnh bbox chính xác hơn.

Loss train giảm là cần thiết nhưng chưa đủ. Loss giảm trong khi validation metric xấu đi là dấu
hiệu overfitting.

### Detection metrics

- **Precision**: trong các box model dự đoán là biển số, bao nhiêu box đúng.
- **Recall**: trong các biển số thật, model tìm được bao nhiêu.
- **mAP@0.5**: average precision khi IoU tối thiểu là 0.5.
- **mAP@0.5:0.95**: trung bình trên nhiều ngưỡng IoU; nghiêm khắc hơn về độ khít bbox.

Ví dụ:

- Precision cao, recall thấp: model thận trọng, bỏ sót nhiều biển.
- Recall cao, precision thấp: tìm được biển nhưng tạo nhiều false positive.
- mAP@0.5 cao nhưng mAP@0.5:0.95 thấp: tìm đúng vật thể nhưng bbox chưa khít.

Không kết luận chỉ từ một epoch. Quan sát xu hướng nhiều epoch và checkpoint `best.pt`.

## Nếu áp dụng cho bài khác

### Detection nhiều class

Thay:

- Dataset labels và `names` trong dataset YAML.
- Số class tự được Ultralytics đọc từ `names`.
- Báo cáo per-class precision, recall và mAP.

### Vật thể rất nhỏ

Cân nhắc:

- Tăng `imgsz`.
- Dùng tiling/cropping.
- Tăng dữ liệu small objects.
- Giảm augmentation làm vật thể nhỏ thêm.

Đổi lại: tốn VRAM và thời gian hơn.

### Dataset lớn hơn

Cân nhắc:

- Batch lớn hơn nếu VRAM cho phép.
- Workers nhiều hơn.
- Cache ảnh.
- Cloud GPU.

### Domain khác

Phải audit lại:

- Class distribution.
- Box size/aspect/location.
- Duplicate leakage.
- Split theo video, người dùng, thiết bị hoặc session.
- Augmentation phù hợp vật lý của domain.

Không sao chép augmentation của biển số sang ảnh y tế hoặc tài liệu mà không kiểm chứng.

## So sánh lựa chọn model

### YOLO

Ưu điểm:

- Train và inference nhanh.
- Pipeline đơn giản.
- Hỗ trợ export ONNX tốt.
- Phù hợp realtime và project end-to-end.

Nhược điểm:

- Vật thể rất nhỏ vẫn khó.
- Kết quả nhạy với resolution và augmentation.
- API/framework thay đổi tương đối nhanh.

### Faster R-CNN

Ưu điểm:

- Two-stage detector, thường mạnh về localization và vật thể khó.
- Dễ phân tích region proposals.

Nhược điểm:

- Chậm hơn.
- Pipeline nặng hơn.
- Khó đạt latency realtime trên CPU.

### DETR và các biến thể

Ưu điểm:

- Kiến trúc transformer end-to-end.
- Không phụ thuộc NMS theo cách detector truyền thống.

Nhược điểm:

- Thường cần nhiều compute/data hơn.
- Baseline và deployment phức tạp hơn YOLO cho project này.

### YOLO11n so với s/m/l

- `n`: nhỏ, nhanh, phù hợp baseline và GPU 4 GB.
- `s`: chính xác hơn nhưng chậm và tốn VRAM hơn.
- `m/l`: chỉ hợp lý khi baseline cho thấy capacity là bottleneck và có GPU đủ mạnh.

Không chọn model lớn trước khi chứng minh model nhỏ đang underfit.

## Checklist trước khi bạn chạy qua đêm

```powershell
conda activate HariAI
python scripts/train_detection.py --config configs/detection-baseline.yaml --check-only
nvidia-smi
python scripts/train_detection.py --config configs/detection-baseline.yaml
```

Trong lúc train:

- Không đóng terminal.
- Không chạy game hoặc workload GPU khác.
- Theo dõi nhiệt độ và VRAM bằng `nvidia-smi`.
- Không sửa dataset/config giữa run.
- Nếu bị ngắt, dùng `last.pt` để resume.
