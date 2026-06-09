# app.py — Vietnamese ABSA PhoBERT Streamlit Demo (Nhóm 9)

import os
import sys
import csv
import pandas as pd
from datetime import datetime
import streamlit as st

# ── Cấu hình trang ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ABSA PhoBERT | Phân tích cảm xúc F&B",
    page_icon="🍜",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS tùy chỉnh ────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Nền tổng thể ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255,255,255,0.1);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* ── Header Hero ── */
.hero-header {
    background: linear-gradient(135deg, rgba(99,102,241,0.3) 0%, rgba(168,85,247,0.3) 100%);
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    text-align: center;
    backdrop-filter: blur(10px);
}
.hero-header h1 {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem 0;
}
.hero-header p {
    color: #94a3b8;
    font-size: 1rem;
    margin: 0;
}

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #94a3b8 !important;
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 500;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
}

/* ── Heading labels ── */
.stMarkdown h4,
.stMarkdown h5 {
    color: #ffffff !important;
}

/* ── Sample prompt text ── */
.stMarkdown p {
    color: #ffffff !important;
}

/* ── Text area ── */
.stTextArea textarea {
    background: rgba(255,255,255,0.98) !important;
    border: 1px solid rgba(99,102,241,0.4) !important;
    color: #111827 !important;
    border-radius: 12px !important;
    font-size: 1rem !important;
}
.stTextArea textarea::placeholder {
    color: #6b7280 !important;
}
.stTextArea textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.2) !important;
}

/* ── Nút bấm ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.4) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99,102,241,0.5) !important;
}

/* ── Card aspect ── */
.aspect-card {
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    transition: transform 0.2s, box-shadow 0.2s;
    border: 1px solid transparent;
}
.aspect-card:hover { transform: translateY(-3px); }

.card-pos {
    background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(52,211,153,0.1));
    border-color: rgba(16,185,129,0.4);
}
.card-neg {
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(248,113,113,0.1));
    border-color: rgba(239,68,68,0.4);
}
.card-neu {
    background: linear-gradient(135deg, rgba(245,158,11,0.15), rgba(251,191,36,0.1));
    border-color: rgba(245,158,11,0.4);
}
.card-none {
    background: rgba(255,255,255,0.03);
    border-color: rgba(255,255,255,0.08);
    opacity: 0.55;
}

.card-title {
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 0.3rem;
}
.card-sentiment {
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 0.4rem;
}
.card-pos .card-sentiment  { color: #34d399; }
.card-neg .card-sentiment  { color: #f87171; }
.card-neu .card-sentiment  { color: #fbbf24; }
.card-none .card-sentiment { color: #64748b; }

.card-conf {
    font-size: 0.78rem;
    color: #64748b;
}

/* ── Progress bar confidence ── */
.conf-bar-wrap {
    background: rgba(255,255,255,0.08);
    border-radius: 99px;
    height: 5px;
    margin-top: 6px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.5s ease;
}
.bar-pos  { background: linear-gradient(90deg, #10b981, #34d399); }
.bar-neg  { background: linear-gradient(90deg, #ef4444, #f87171); }
.bar-neu  { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
.bar-none { background: #334155; }

/* ── Metric cards ── */
.metric-box {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px;
    padding: 1.2rem;
    text-align: center;
}
.metric-box .metric-val {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-box .metric-lbl {
    font-size: 0.8rem;
    color: #64748b;
    margin-top: 0.2rem;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.08) !important; }

/* ── Alert boxes ── */
.info-box {
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    color: #a5b4fc;
    font-size: 0.9rem;
}

/* ── Spinner text & icon ── */
[data-testid="stSpinner"] {
    color: #ffffff !important;
}
[data-testid="stSpinner"] * {
    color: #ffffff !important;
}

</style>
""", unsafe_allow_html=True)

# ── Hằng số ─────────────────────────────────────────────────────────────────
MODEL_READY = os.path.exists("models/best_model/model.safetensors") and \
              os.path.getsize("models/best_model/model.safetensors") > 1_000_000

PREDICTIONS_CSV = "outputs/predictions.csv"
LABEL_COLS = [
    'AMBIENCE#GENERAL', 'DRINKS#PRICES', 'DRINKS#QUALITY', 'DRINKS#STYLE&OPTIONS',
    'FOOD#PRICES', 'FOOD#QUALITY', 'FOOD#STYLE&OPTIONS', 'LOCATION#GENERAL',
    'RESTAURANT#GENERAL', 'RESTAURANT#MISCELLANEOUS', 'RESTAURANT#PRICES', 'SERVICE#GENERAL'
]
ASPECT_VI_MAP = {
    'AMBIENCE#GENERAL':          'Không gian & Bầu không khí',
    'DRINKS#PRICES':             'Giá cả đồ uống',
    'DRINKS#QUALITY':            'Chất lượng đồ uống',
    'DRINKS#STYLE&OPTIONS':      'Menu & Đa dạng đồ uống',
    'FOOD#PRICES':               'Giá cả đồ ăn',
    'FOOD#QUALITY':              'Chất lượng đồ ăn',
    'FOOD#STYLE&OPTIONS':        'Menu & Đa dạng đồ ăn',
    'LOCATION#GENERAL':          'Vị trí & Địa điểm',
    'RESTAURANT#GENERAL':        'Đánh giá chung',
    'RESTAURANT#MISCELLANEOUS':  'Tiện ích khác (Wifi, giữ xe…)',
    'RESTAURANT#PRICES':         'Mức giá chung nhà hàng',
    'SERVICE#GENERAL':           'Chất lượng phục vụ',
}
SENTIMENT_MAP   = {0: "Không đề cập 😶", 1: "Tích cực 😊", 2: "Tiêu cực 😡", 3: "Trung lập 😐"}
SENTIMENT_CLASS = {0: "card-none", 1: "card-pos", 2: "card-neg", 3: "card-neu"}
BAR_CLASS       = {0: "bar-none",  1: "bar-pos",  2: "bar-neg",  3: "bar-neu"}

# ── Load model (cache) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_predictor():
    from utils.predictor import ABSAPredictor
    return ABSAPredictor()

# ── Helpers ──────────────────────────────────────────────────────────────────
def render_aspect_card(item: dict):
    sid   = item["sentiment_id"]
    cc    = SENTIMENT_CLASS[sid]
    bc    = BAR_CLASS[sid]
    conf  = item["confidence"]
    st.markdown(f"""
    <div class="aspect-card {cc}">
        <div class="card-title">{item['aspect_en']}</div>
        <div class="card-sentiment">{item['sentiment']}</div>
        <div class="card-conf">Độ tin cậy: <strong>{conf}%</strong></div>
        <div class="conf-bar-wrap">
            <div class="conf-bar-fill {bc}" style="width:{conf}%"></div>
        </div>
        <div class="card-conf" style="margin-top:4px">{item['aspect_vi']}</div>
    </div>
    """, unsafe_allow_html=True)


def save_to_csv(review_text: str, results_full: list):
    """Ghi một dòng kết quả vào predictions.csv."""
    row = {
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "review_text":  review_text,
    }
    aspect_map = {r["aspect_en"]: r["sentiment_id"] for r in results_full}
    for col in LABEL_COLS:
        row[col] = aspect_map.get(col, 0)
    row["num_mentioned"] = sum(1 for r in results_full if r["sentiment_id"] != 0)

    file_exists = os.path.isfile(PREDICTIONS_CSV) and os.path.getsize(PREDICTIONS_CSV) > 0
    with open(PREDICTIONS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🍜 ABSA PhoBERT")
    st.markdown("**Nhóm 9** — Trí tuệ nhân tạo")
    st.divider()

    st.markdown("### 🏗️ Mô hình")
    st.markdown("- **Backbone**: PhoBERT-Large")
    st.markdown("- **Task**: Multi-Head ABSA")
    st.markdown("- **Khía cạnh**: 12 aspects")
    st.markdown("- **Nhãn**: 4 classes")
    st.divider()

    if MODEL_READY:
        st.success("Model đã load sẵn sàng", icon="✅")
    else:
        st.warning("⚠️ model.safetensors chưa có.\nExport từ Colab trước khi predict thật.", icon="⚠️")
        st.info("Demo UI đang ở **chế độ thử nghiệm** (mock mode).", icon="ℹ️")

    st.divider()
    st.markdown("### 🎭 Nhãn cảm xúc")
    st.markdown("- 😶 **Không đề cập** (0)")
    st.markdown("- 😊 **Tích cực** (1)")
    st.markdown("- 😡 **Tiêu cực** (2)")
    st.markdown("- 😐 **Trung lập** (3)")


# ════════════════════════════════════════════════════════════════════════════
#  HERO HEADER
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-header">
    <h1>🍜 Phân tích Cảm xúc Đa khía cạnh</h1>
    <p>Vietnamese Aspect-Based Sentiment Analysis · PhoBERT-Large · 12 Aspects · F&B Domain</p>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  TABS
# ════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🔍 Phân tích đơn", "📋 Phân tích hàng loạt", "📊 Lịch sử & Thống kê"])


# ────────────────────────────────────────────────────────────────────────────
#  TAB 1 — Phân tích đơn
# ────────────────────────────────────────────────────────────────────────────
with tab1:
    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown("#### ✍️ Nhập đánh giá tiếng Việt")

        # Câu mẫu nhanh — đặt TRƯỚC text_area để preset kịp áp dụng
        st.markdown("**💡 Thử ngay với đánh giá mẫu:**")
        samples = {
            "🍕 Tích cực tổng thể": "Quán này thực sự tuyệt vời! Đồ ăn ngon, không gian đẹp và nhân viên phục vụ rất nhiệt tình.",
            "😤 Tiêu cực dịch vụ":  "Thức ăn bình thường nhưng nhân viên thái độ rất tệ, chờ mãi không được phục vụ.",
            "🤔 Hỗn hợp":           "Đồ uống ngon và giá cả hợp lý, nhưng chỗ ngồi hơi chật và ồn ào.",
        }
        for label, text in samples.items():
            if st.button(label, key=f"sample_{label}"):
                # Gán vào key PHỤ (_preset) + cờ _auto_run để tự phân tích luôn
                st.session_state["_preset_review"] = text
                st.session_state["_auto_run"] = True
                st.rerun()

        # Đọc preset (nếu có) làm giá trị mặc định cho text_area, rồi xóa đi
        preset_value = st.session_state.pop("_preset_review", "")
        review_text = st.text_area(
            label="",
            value=preset_value,
            placeholder="Ví dụ: Quán có không gian rộng, đồ ăn ngon nhưng giá hơi cao và nhân viên phục vụ chậm...",
            height=160,
            key="single_review",
        )

        run_btn = st.button("🚀 Phân tích ngay", key="run_single", use_container_width=True)

    # Đọc cờ auto_run (từ nút mẫu) — xóa ngay sau khi đọc để không lặp vô hạn
    auto_run = st.session_state.pop("_auto_run", False)

    with col_result:
        st.markdown("#### 📊 Kết quả phân tích")
        placeholder = st.empty()

        if not (run_btn or auto_run) or not review_text.strip():
            placeholder.markdown("""
            <div class="info-box">
                ← Nhập câu review và nhấn <strong>"Phân tích ngay"</strong> để xem kết quả trên 12 khía cạnh.
            </div>
            """, unsafe_allow_html=True)
        else:
            with st.spinner("🔄 Đang phân tích..."):
                if MODEL_READY:
                    predictor = load_predictor()
                    results_full = predictor.predict_full(review_text)
                else:
                    # ── MOCK MODE khi chưa có model ──────────────────────
                    import random
                    random.seed(hash(review_text) % 999)
                    MOCK_NOTE = True
                    results_full = []
                    for asp in LABEL_COLS:
                        sid = random.choices([0,1,2,3], weights=[0.4,0.25,0.2,0.15])[0]
                        probs_fake = [random.random() for _ in range(4)]
                        probs_fake[sid] += 2
                        total = sum(probs_fake)
                        probs_fake = [p/total for p in probs_fake]
                        conf = round(probs_fake[sid]*100, 2)
                        results_full.append({
                            "aspect_en":    asp,
                            "aspect_vi":    ASPECT_VI_MAP[asp],
                            "sentiment":    SENTIMENT_MAP[sid],
                            "sentiment_id": sid,
                            "confidence":   conf,
                            "mentioned":    sid != 0,
                        })

            # Metrics tổng quan
            mentioned = [r for r in results_full if r["sentiment_id"] != 0]
            pos_count = sum(1 for r in results_full if r["sentiment_id"] == 1)
            neg_count = sum(1 for r in results_full if r["sentiment_id"] == 2)
            neu_count = sum(1 for r in results_full if r["sentiment_id"] == 3)

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f'<div class="metric-box"><div class="metric-val">{len(mentioned)}</div><div class="metric-lbl">Khía cạnh được đề cập</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-box"><div class="metric-val" style="background:linear-gradient(135deg,#10b981,#34d399);-webkit-background-clip:text;-webkit-text-fill-color:transparent">{pos_count}</div><div class="metric-lbl">Tích cực 😊</div></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="metric-box"><div class="metric-val" style="background:linear-gradient(135deg,#ef4444,#f87171);-webkit-background-clip:text;-webkit-text-fill-color:transparent">{neg_count}</div><div class="metric-lbl">Tiêu cực 😡</div></div>', unsafe_allow_html=True)
            with m4:
                st.markdown(f'<div class="metric-box"><div class="metric-val" style="background:linear-gradient(135deg,#f59e0b,#fbbf24);-webkit-background-clip:text;-webkit-text-fill-color:transparent">{neu_count}</div><div class="metric-lbl">Trung lập 😐</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            if not MODEL_READY:
                st.warning("⚠️ **Mock mode** — Kết quả ngẫu nhiên do chưa có model.safetensors. Export từ Colab để chạy thật.", icon="⚠️")

            # Hiển thị 12 card theo 2 cột
            c1, c2 = st.columns(2)
            for idx, item in enumerate(results_full):
                with (c1 if idx % 2 == 0 else c2):
                    render_aspect_card(item)

            # Lưu kết quả vào CSV
            save_to_csv(review_text, results_full)
            st.toast("✅ Đã lưu kết quả vào predictions.csv", icon="💾")


# ────────────────────────────────────────────────────────────────────────────
#  TAB 2 — Phân tích hàng loạt
# ────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("#### 📂 Upload file CSV để phân tích nhiều câu review")
    st.markdown("""
    <div class="info-box">
        File CSV cần có cột <code>review</code> chứa các câu review tiếng Việt. Mỗi dòng sẽ được phân tích tự động.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Template download
    template_csv = "review\nQuán này thực sự tuyệt vời! Đồ ăn ngon và nhân viên nhiệt tình.\nGiá hơi cao nhưng không gian đẹp.\nNhân viên phục vụ rất chậm và thái độ không tốt.\n"
    st.download_button(
        label="📥 Tải file mẫu CSV",
        data=template_csv.encode("utf-8"),
        file_name="reviews_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("", type=["csv"], key="batch_upload")

    if uploaded_file:
        try:
            df_input = pd.read_csv(uploaded_file)
            if "review" not in df_input.columns:
                st.error("❌ File CSV phải có cột tên là `review`.")
            else:
                st.success(f"✅ Đã tải {len(df_input)} câu review.")
                st.dataframe(df_input.head(5), use_container_width=True)

                if st.button("🚀 Bắt đầu phân tích hàng loạt", key="run_batch"):
                    if not MODEL_READY:
                        st.warning("⚠️ Đang chạy Mock mode (chưa có model thật).", icon="⚠️")

                    progress = st.progress(0, text="Đang xử lý...")
                    results_rows = []

                    predictor = load_predictor() if MODEL_READY else None

                    for i, row in df_input.iterrows():
                        text = str(row["review"])
                        progress.progress((i + 1) / len(df_input), text=f"Đang xử lý {i+1}/{len(df_input)}...")

                        if predictor:
                            full = predictor.predict_full(text)
                        else:
                            import random
                            random.seed(hash(text) % 999)
                            full = []
                            for asp in LABEL_COLS:
                                sid = random.choices([0,1,2,3], weights=[0.4,0.25,0.2,0.15])[0]
                                full.append({"aspect_en": asp, "sentiment_id": sid,
                                             "sentiment": SENTIMENT_MAP[sid], "confidence": 0.0,
                                             "aspect_vi": ASPECT_VI_MAP[asp], "mentioned": sid!=0})

                        result_row = {"review": text}
                        for r in full:
                            result_row[r["aspect_en"]] = SENTIMENT_MAP[r["sentiment_id"]]
                        result_row["num_mentioned"] = sum(1 for r in full if r["sentiment_id"] != 0)
                        results_rows.append(result_row)
                        save_to_csv(text, full)

                    progress.empty()
                    df_out = pd.DataFrame(results_rows)
                    st.success(f"✅ Hoàn thành! Đã phân tích {len(df_out)} câu.")
                    st.dataframe(df_out, use_container_width=True)

                    csv_bytes = df_out.to_csv(index=False, encoding="utf-8").encode("utf-8")
                    st.download_button(
                        label="📥 Tải kết quả CSV",
                        data=csv_bytes,
                        file_name=f"absa_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                    )
        except Exception as e:
            st.error(f"❌ Lỗi đọc file: {e}")


# ────────────────────────────────────────────────────────────────────────────
#  TAB 3 — Lịch sử & Thống kê
# ────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("#### 📊 Lịch sử dự đoán")

    if not os.path.isfile(PREDICTIONS_CSV) or os.path.getsize(PREDICTIONS_CSV) < 10:
        st.markdown("""
        <div class="info-box">
            Chưa có dữ liệu lịch sử. Hãy phân tích một vài câu review ở tab <strong>Phân tích đơn</strong> trước.
        </div>
        """, unsafe_allow_html=True)
    else:
        df_hist = pd.read_csv(PREDICTIONS_CSV, encoding="utf-8")
        df_hist = df_hist.sort_values("timestamp", ascending=False).reset_index(drop=True)

        # Metrics tổng quan
        total_reviews = len(df_hist)
        avg_mentioned = df_hist["num_mentioned"].mean() if "num_mentioned" in df_hist else 0

        h1, h2, h3 = st.columns(3)
        with h1:
            st.markdown(f'<div class="metric-box"><div class="metric-val">{total_reviews}</div><div class="metric-lbl">Tổng câu đã phân tích</div></div>', unsafe_allow_html=True)
        with h2:
            st.markdown(f'<div class="metric-box"><div class="metric-val">{avg_mentioned:.1f}</div><div class="metric-lbl">TB khía cạnh / câu</div></div>', unsafe_allow_html=True)
        with h3:
            if "num_mentioned" in df_hist:
                max_mentioned = df_hist["num_mentioned"].max()
                st.markdown(f'<div class="metric-box"><div class="metric-val">{max_mentioned}</div><div class="metric-lbl">Max khía cạnh trong 1 câu</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Biểu đồ phân bố sentiment theo aspect
        st.markdown("##### 📈 Phân bố nhãn theo khía cạnh")
        aspect_cols = [c for c in LABEL_COLS if c in df_hist.columns]
        if aspect_cols:
            dist_data = {}
            for col in aspect_cols:
                counts = df_hist[col].value_counts()
                dist_data[ASPECT_VI_MAP.get(col, col)] = {
                    "Tích cực": int(counts.get(1, 0)),
                    "Tiêu cực": int(counts.get(2, 0)),
                    "Trung lập": int(counts.get(3, 0)),
                }
            df_dist = pd.DataFrame(dist_data).T
            st.bar_chart(df_dist, color=["#34d399", "#f87171", "#fbbf24"])

        st.markdown("<br>", unsafe_allow_html=True)

        # Bảng lịch sử
        st.markdown("##### 🗒️ Chi tiết lịch sử")
        display_cols = ["timestamp", "review_text", "num_mentioned"] + aspect_cols[:6]
        st.dataframe(
            df_hist[display_cols] if all(c in df_hist.columns for c in display_cols) else df_hist,
            use_container_width=True,
            height=350,
        )

        # Export
        st.download_button(
            label="📥 Export toàn bộ lịch sử CSV",
            data=df_hist.to_csv(index=False, encoding="utf-8").encode("utf-8"),
            file_name=f"predictions_history_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
