# 🍔 Ứng dụng Phân tích Cảm xúc Đa khía cạnh (ABSA) với PhoBERT-Large

Ứng dụng dự đoán trạng thái cảm xúc người dùng trên 12 khía cạnh thuộc lĩnh vực F&B, sử dụng mô hình học sâu tối ưu PhoBERT-Large Multi-Head.

---

# 📁 Cấu trúc thư mục dự án

```text
phobert_absa_streamlit/
│
├── app.py                # Giao diện chính chạy bằng Streamlit
├── models/
│   ├── best_model/       # Nơi đặt file trọng số và file cấu hình (.safetensors, .json)
│   └── phobert_model.py     # Định nghĩa class kiến trúc PhoBERT_ABSA_MultiHead
├── utils/                   
│   ├── preprocessing.py     # Hàm làm sạch và phân từ tiếng Việt bằng PyVi
│   ├── predictor.py         # Code xử lý nạp mô hình và chạy dự đoán (Inference)
│   └── constants.py         # Khai báo siêu tham số, mảng khía cạnh và từ điển ánh xạ
├── requirements.txt         # Danh sách các thư viện cần cài đặt
└── README.md                # Tài liệu hướng dẫn triển khai hệ thống