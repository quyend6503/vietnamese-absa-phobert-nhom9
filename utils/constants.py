# CẤU HÌNH SIÊU THAM SỐ TỪ COLAB
# Tên mô hình gốc được sử dụng trong notebook huấn luyện
MODEL_NAME = "vinai/phobert-large"
# Đường dẫn cục bộ tới thư mục chứa bộ trọng số sau khi train (dùng cho predictor.py)
MODEL_PATH = "models/best_model"
# Độ dài tối đa của câu
MAX_LEN = 256
# Cấu hình kiến trúc mạng đa nhiệm
NUM_ASPECTS = 12
NUM_CLASSES = 4

# DANH SÁCH 12 KHÍA CẠNH
LABEL_COLS = [
    'AMBIENCE#GENERAL', 
    'DRINKS#PRICES', 
    'DRINKS#QUALITY', 
    'DRINKS#STYLE&OPTIONS',
    'FOOD#PRICES', 
    'FOOD#QUALITY', 
    'FOOD#STYLE&OPTIONS', 
    'LOCATION#GENERAL',
    'RESTAURANT#GENERAL', 
    'RESTAURANT#MISCELLANEOUS', 
    'RESTAURANT#PRICES', 
    'SERVICE#GENERAL'
]

# ÁNH XẠ KẾT QUẢ ĐẦU RA
# Đồng bộ chính xác với từ điển sentiments_map = {0: 'Không đề cập', 1: 'Tích cực', ...} lúc train
SENTIMENT_MAP = {
    0: "Không đề cập 😶",
    1: "Tích cực 😊",
    2: "Tiêu cực 😡",
    3: "Trung lập 😐"
}

# Bảng dịch giao diện Tiếng Việt
ASPECT_VI_MAP = {
    'AMBIENCE#GENERAL': 'Không gian & Bầu không khí',
    'DRINKS#PRICES': 'Giá cả đồ uống',
    'DRINKS#QUALITY': 'Chất lượng đồ uống',
    'DRINKS#STYLE&OPTIONS': 'Menu & Sự đa dạng đồ uống',
    'FOOD#PRICES': 'Giá cả đồ ăn',
    'FOOD#QUALITY': 'Chất lượng đồ ăn',
    'FOOD#STYLE&OPTIONS': 'Menu & Sự đa dạng đồ ăn',
    'LOCATION#GENERAL': 'Vị trí & Địa điểm nhà hàng',
    'RESTAURANT#GENERAL': 'Đánh giá chung về nhà hàng',
    'RESTAURANT#MISCELLANEOUS': 'Tiện ích khác (Giữ xe, Wifi, không gian...)',
    'RESTAURANT#PRICES': 'Mức giá chung của nhà hàng',
    'SERVICE#GENERAL': 'Chất lượng phục vụ của nhân viên'
}