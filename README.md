# 🍔 Ứng dụng Phân tích Cảm xúc Đa khía cạnh (ABSA) với PhoBERT-Large

# Vietnamese ABSA using PhoBERT

Ứng dụng dự đoán trạng thái cảm xúc người dùng trên 12 khía cạnh thuộc lĩnh vực F&B, sử dụng mô hình học sâu tối ưu PhoBERT-Large Multi-Head.

## Giới thiệu

Dự án xây dựng hệ thống **Aspect-Based Sentiment Analysis (ABSA)** cho các đánh giá trong lĩnh vực **dịch vụ ăn uống (F&B)** sử dụng mô hình **PhoBERT Large** kết hợp kiến trúc **Multi-Head Classification**.

Hệ thống có khả năng:

* Xác định 12 khía cạnh (Aspect) trong một câu đánh giá.
* Phân loại cảm xúc cho từng khía cạnh:

  * Tích cực (Positive)
  * Tiêu cực (Negative)
  * Trung lập (Neutral)
  * Không đề cập (None)
* Hiển thị kết quả thông qua giao diện Streamlit.

---

## Công nghệ sử dụng

* Python 3.10+
* PhoBERT Large
* PyTorch
* Transformers
* Streamlit
* PyVi
* Safetensors

---

## Cấu trúc thư mục

```text
vietnamese-absa-phobert-nhom9/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── phobert_model.py
│   └── best_model/
│       ├── model.safetensors
│       └── config_absa.json
│
├── utils/
│   ├── constants.py
│   ├── predictor.py
│   └── preprocessing.py
│
├── outputs/
│   └── predictions.csv
│
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
├── vocab.txt
└── config.json
```

---

## Yêu cầu cài đặt

### 1. Cài Python

Khuyến nghị sử dụng:

```text
Python 3.10 hoặc Python 3.11
```

Kiểm tra phiên bản:

```bash
python --version
```

---

### 2. Cài Git

Kiểm tra:

```bash
git --version
```

---

### 3. Cài Git LFS

Dự án sử dụng file mô hình lớn:

```text
model.safetensors (~1.4GB)
```

Cài Git LFS:

https://git-lfs.com

Sau khi cài đặt:

```bash
git lfs install
```

Kiểm tra:

```bash
git lfs version
```

---

## Clone dự án

```bash
git clone https://github.com/quyend6503/vietnamese-absa-phobert-nhom9.git
```

Di chuyển vào thư mục dự án:

```bash
cd vietnamese-absa-phobert-nhom9
```

---

## Tải mô hình từ Git LFS

Sau khi clone:

```bash
git lfs pull
```

Kiểm tra file:

```text
models/best_model/model.safetensors
```

Dung lượng khoảng:

```text
1.4 GB
```

Nếu file chỉ vài KB thì Git LFS chưa tải mô hình thành công.

---

## Cài đặt thư viện

```bash
pip install -r requirements.txt
```

Nếu thiếu thư viện:

```bash
pip install torch
pip install transformers
pip install streamlit
pip install pyvi
pip install pandas
pip install numpy
pip install scikit-learn
pip install safetensors
pip install sentencepiece
```

---

## Chạy ứng dụng

Tại thư mục gốc dự án:

```bash
streamlit run app.py
```

Sau khi chạy thành công:

```text
Local URL: http://localhost:8501
```

Mở trình duyệt và truy cập địa chỉ trên.

---

## Các khía cạnh được phân tích

1. AMBIENCE#GENERAL
2. DRINKS#PRICES
3. DRINKS#QUALITY
4. DRINKS#STYLE&OPTIONS
5. FOOD#PRICES
6. FOOD#QUALITY
7. FOOD#STYLE&OPTIONS
8. LOCATION#GENERAL
9. RESTAURANT#GENERAL
10. RESTAURANT#MISCELLANEOUS
11. RESTAURANT#PRICES
12. SERVICE#GENERAL

---

## Các nhãn cảm xúc

| ID | Nhãn         |
| -- | ------------ |
| 0  | Không đề cập |
| 1  | Tích cực     |
| 2  | Tiêu cực     |
| 3  | Trung lập    |

---

## Thành viên nhóm

* Đặng Thị Quyên            -   26A4042136
* Nguyễn Thị Ngọc Châm      -   26A4041216
* Vàng Thị Nguyên           -   26A4042121
* Nguyễn Thị Phương Thảo    -   26A4042141
* Nguyễn Thu Trang          -   26A4042152

---

## Lưu ý

* Không xóa thư mục `models/best_model`.
* Không đổi tên file `model.safetensors`.
* Luôn chạy `git lfs pull` sau khi clone dự án.
* Nếu gặp lỗi thiếu thư viện, cài lại bằng:

```bash
pip install -r requirements.txt
```

* Nếu gặp lỗi mô hình, kiểm tra sự tồn tại của:

```text
models/best_model/model.safetensors
```