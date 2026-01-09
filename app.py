import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import time

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .fraud-alert {
        background-color: #ff4b4b;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .normal-alert {
        background-color: #00cc00;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL AND SCALER
# ============================================================================
@st.cache_resource
def load_models():
    """Load trained model and scaler"""
    try:
        model = joblib.load('models/optimized_rf.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"❌ Lỗi khi load model: {str(e)}")
        st.info("Vui lòng đảm bảo các file sau tồn tại:\n- models/optimized_rf.pkl\n- models/scaler.pkl")
        st.stop()

# ============================================================================
# FRAUD DETECTOR CLASS
# ============================================================================
class FraudDetector:
    def __init__(self, model, scaler, threshold=0.35):
        self.model = model
        self.scaler = scaler
        self.threshold = threshold
        self.feature_names = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
                             'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18',
                             'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 
                             'V28', 'Amount_log']
    
    def preprocess(self, transaction):
        """Preprocess transaction data"""
        # Create DataFrame
        df = pd.DataFrame([transaction])
        
        # Log transform Amount
        if 'Amount' in df.columns:
            df['Amount_log'] = np.log1p(df['Amount'])
            df = df.drop('Amount', axis=1)
        
        # Ensure all features exist
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0.0
        
        # Reorder columns
        df = df[self.feature_names]
        
        # Scale
        X_scaled = self.scaler.transform(df)
        
        return X_scaled
    
    def predict_single(self, transaction):
        """Predict single transaction"""
        X = self.preprocess(transaction)
        
        # Get prediction and probability
        prediction = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        fraud_prob = proba[1]
        
        # Apply custom threshold
        prediction_custom = 1 if fraud_prob >= self.threshold else 0
        
        return {
            'prediction': 'FRAUD' if prediction_custom == 1 else 'NORMAL',
            'fraud_probability': fraud_prob,
            'normal_probability': proba[0],
            'confidence': max(proba),
            'threshold': self.threshold
        }

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_risk_level(fraud_prob):
    """Get risk level based on fraud probability"""
    if fraud_prob >= 0.8:
        return "⛔ RỦI RO CỰC CAO", "#ff0000"
    elif fraud_prob >= 0.5:
        return "⚠️ RỦI RO CAO", "#ff6600"
    elif fraud_prob >= 0.3:
        return "🟡 RỦI RO TRUNG BÌNH", "#ffcc00"
    else:
        return "✅ RỦI RO THẤP", "#00cc00"

def create_probability_gauge(fraud_prob, threshold=0.35):
    """Create gauge chart for fraud probability"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = fraud_prob * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Xác suất gian lận (%)"},
        delta = {'reference': threshold * 100},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 50], 'color': "yellow"},
                {'range': [50, 80], 'color': "orange"},
                {'range': [80, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Header
    st.markdown('<div class="main-header">💳 HỆ THỐNG PHÁT HIỆN GIAN LẬN THẺ TÍN DỤNG</div>', 
                unsafe_allow_html=True)
    
    # Load models
    with st.spinner('🔄 Đang load mô hình...'):
        model, scaler = load_models()
        detector = FraudDetector(model, scaler)
    
    # Sidebar
    st.sidebar.title("⚙️ CÀI ĐẶT")
    
    # Threshold adjustment
    threshold = st.sidebar.slider(
        "Ngưỡng phát hiện fraud",
        min_value=0.0,
        max_value=1.0,
        value=0.35,
        step=0.05,
        help="Ngưỡng càng thấp, càng nhạy cảm với fraud"
    )
    detector.threshold = threshold
    
    # Mode selection
    mode = st.sidebar.radio(
        "Chọn chế độ:",
        ["🔍 Dự đoán đơn lẻ", "📊 Dự đoán hàng loạt", "📁 Upload file CSV"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📖 Hướng dẫn sử dụng")
    st.sidebar.info("""
    **Dự đoán đơn lẻ:** Nhập thông tin giao dịch thủ công
    
    **Dự đoán hàng loạt:** Sử dụng dữ liệu test có sẵn
    
    **Upload file:** Tải lên file CSV của bạn
    """)
    
    # ========================================================================
    # MODE 1: Single Prediction
    # ========================================================================
    if mode == "🔍 Dự đoán đơn lẻ":
        st.header("🔍 Dự đoán giao dịch đơn lẻ")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📝 Thông tin giao dịch")
            
            time_input = st.number_input(
                "Time (giây từ giao dịch đầu)",
                min_value=0.0,
                value=406.0,
                help="Thời gian tính từ giao dịch đầu tiên"
            )
            
            amount_input = st.number_input(
                "Amount (số tiền giao dịch, $)",
                min_value=0.0,
                value=149.62,
                help="Số tiền giao dịch"
            )
            
            st.markdown("---")
            st.subheader("🔢 V Features quan trọng")
            st.caption("Các features được tạo từ PCA. Để trống = 0")
            
            # Important V features
            v17 = st.number_input("V17", value=-2.83)
            v14 = st.number_input("V14", value=-4.29)
            v12 = st.number_input("V12", value=-2.90)
            v10 = st.number_input("V10", value=-2.77)
            v16 = st.number_input("V16", value=-1.14)
        
        with col2:
            st.subheader("🎯 Kết quả dự đoán")
            
            # Create transaction dict
            transaction = {
                'Time': time_input,
                'Amount': amount_input,
                'V17': v17, 'V14': v14, 'V12': v12,
                'V10': v10, 'V16': v16
            }
            
            # Add other V features = 0
            for i in range(1, 29):
                feature = f'V{i}'
                if feature not in transaction:
                    transaction[feature] = 0.0
            
            if st.button("🚀 DỰ ĐOÁN", type="primary", use_container_width=True):
                with st.spinner('Đang phân tích...'):
                    time.sleep(0.5)  # Animation
                    result = detector.predict_single(transaction)
                
                # Display result
                if result['prediction'] == 'FRAUD':
                    st.markdown(
                        f'<div class="fraud-alert">⛔ CẢNH BÁO: GIAO DỊCH GIAN LẬN</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="normal-alert">✅ GIAO DỊCH HỢP LỆ</div>',
                        unsafe_allow_html=True
                    )
                
                st.markdown("---")
                
                # Metrics
                col_m1, col_m2, col_m3 = st.columns(3)
                
                with col_m1:
                    st.metric(
                        "Xác suất Fraud",
                        f"{result['fraud_probability']:.2%}",
                        delta=f"{(result['fraud_probability'] - threshold):.2%}"
                    )
                
                with col_m2:
                    st.metric(
                        "Độ tin cậy",
                        f"{result['confidence']:.2%}"
                    )
                
                with col_m3:
                    risk_level, risk_color = get_risk_level(result['fraud_probability'])
                    st.markdown(f"""
                    <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid {risk_color};">
                        <p style="color: #888; font-size: 0.875rem; margin: 0;">Mức độ rủi ro</p>
                        <p style="color: {risk_color}; font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0 0 0;">{risk_level}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Gauge chart
                st.plotly_chart(
                    create_probability_gauge(result['fraud_probability'], detector.threshold),
                    use_container_width=True
                )
                
                # Details
                with st.expander("📊 Chi tiết phân tích"):
                    st.json({
                        'Prediction': result['prediction'],
                        'Fraud Probability': f"{result['fraud_probability']:.4f}",
                        'Normal Probability': f"{result['normal_probability']:.4f}",
                        'Confidence': f"{result['confidence']:.4f}",
                        'Threshold': f"{result['threshold']:.4f}",
                        'Amount': f"${amount_input:.2f}",
                        'Time': f"{time_input:.0f}s"
                    })
    
    # ========================================================================
    # MODE 2: Batch Prediction
    # ========================================================================
    elif mode == "📊 Dự đoán hàng loạt":
        st.header("📊 Dự đoán hàng loạt")
        
        try:
            test_data = pd.read_csv('data/processed/test.csv')
            
            n_samples = st.slider(
                "Số lượng giao dịch muốn test:",
                min_value=5,
                max_value=min(100, len(test_data)),
                value=10
            )
            
            if st.button("🚀 BẮT ĐẦU DỰ ĐOÁN", type="primary"):
                with st.spinner(f'Đang dự đoán {n_samples} giao dịch...'):
                    # Get batch
                    batch = test_data.drop('Class', axis=1).head(n_samples)
                    true_labels = test_data['Class'].head(n_samples)
                    
                    # Predict each
                    results = []
                    progress_bar = st.progress(0)
                    
                    for i in range(n_samples):
                        transaction = batch.iloc[i].to_dict()
                        result = detector.predict_single(transaction)
                        
                        results.append({
                            'Transaction': i + 1,
                            'True_Label': 'FRAUD' if true_labels.iloc[i] == 1 else 'NORMAL',
                            'Prediction': result['prediction'],
                            'Fraud_Prob': result['fraud_probability'],
                            'Confidence': result['confidence'],
                            'Correct': (result['prediction'] == 'FRAUD') == (true_labels.iloc[i] == 1)
                        })
                        
                        progress_bar.progress((i + 1) / n_samples)
                    
                    results_df = pd.DataFrame(results)
                    
                    # Statistics
                    st.success("✅ Hoàn thành dự đoán!")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Tổng giao dịch", n_samples)
                    
                    with col2:
                        fraud_count = sum(results_df['Prediction'] == 'FRAUD')
                        st.metric("Dự đoán FRAUD", fraud_count)
                    
                    with col3:
                        accuracy = results_df['Correct'].mean()
                        st.metric("Accuracy", f"{accuracy:.1%}")
                    
                    with col4:
                        true_fraud = sum(results_df['True_Label'] == 'FRAUD')
                        st.metric("Thực tế FRAUD", true_fraud)
                    
                    # Results table
                    st.subheader("📋 Kết quả chi tiết")
                    
                    # Color code
                    def highlight_row(row):
                        if not row['Correct']:
                            return ['background-color: #ffcccc'] * len(row)
                        elif row['Prediction'] == 'FRAUD':
                            return ['background-color: #ffe6e6'] * len(row)
                        return [''] * len(row)
                    
                    st.dataframe(
                        results_df.style.apply(highlight_row, axis=1),
                        use_container_width=True
                    )
                    
                    # Misclassified
                    misclassified = results_df[~results_df['Correct']]
                    if len(misclassified) > 0:
                        st.warning(f"⚠️ {len(misclassified)} giao dịch dự đoán sai")
                        st.dataframe(misclassified, use_container_width=True)
                    else:
                        st.success("🎉 Tất cả dự đoán đều chính xác!")
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Tải xuống kết quả (CSV)",
                        data=csv,
                        file_name="fraud_detection_results.csv",
                        mime="text/csv"
                    )
        
        except FileNotFoundError:
            st.error("❌ Không tìm thấy file test data!")
            st.info("Vui lòng đảm bảo file 'data/processed/test.csv' tồn tại")
    
    # ========================================================================
    # MODE 3: Upload CSV
    # ========================================================================
    else:
        st.header("📁 Upload file CSV")
        
        st.info("""
        📝 **Yêu cầu file CSV:**
        - Phải chứa các cột: Time, V1-V28, Amount
        - Không cần cột Class (nếu có sẽ bị bỏ qua)
        - Format giống với dữ liệu training
        """)
        
        uploaded_file = st.file_uploader("Chọn file CSV", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                st.success(f"✅ Đã load {len(df)} giao dịch")
                
                # Preview
                with st.expander("👀 Xem trước dữ liệu"):
                    st.dataframe(df.head(10))
                
                if st.button("🚀 DỰ ĐOÁN TẤT CẢ", type="primary"):
                    with st.spinner(f'Đang dự đoán {len(df)} giao dịch...'):
                        results = []
                        progress_bar = st.progress(0)
                        
                        for i in range(len(df)):
                            transaction = df.iloc[i].to_dict()
                            # Remove Class if exists
                            transaction.pop('Class', None)
                            
                            result = detector.predict_single(transaction)
                            
                            results.append({
                                'Transaction': i + 1,
                                'Prediction': result['prediction'],
                                'Fraud_Prob': result['fraud_probability'],
                                'Confidence': result['confidence']
                            })
                            
                            if i % 10 == 0:
                                progress_bar.progress((i + 1) / len(df))
                        
                        progress_bar.progress(1.0)
                        results_df = pd.DataFrame(results)
                        
                        # Add to original df
                        df['Prediction'] = results_df['Prediction']
                        df['Fraud_Probability'] = results_df['Fraud_Prob']
                        df['Confidence'] = results_df['Confidence']
                        
                        st.success("✅ Hoàn thành!")
                        
                        # Statistics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Tổng giao dịch", len(df))
                        
                        with col2:
                            fraud_count = sum(df['Prediction'] == 'FRAUD')
                            st.metric("Dự đoán FRAUD", fraud_count)
                        
                        with col3:
                            fraud_pct = fraud_count / len(df) * 100
                            st.metric("Tỷ lệ FRAUD", f"{fraud_pct:.2f}%")
                        
                        # Results
                        st.subheader("📊 Kết quả")
                        st.dataframe(df, use_container_width=True)
                        
                        # Download
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Tải xuống kết quả (CSV)",
                            data=csv,
                            file_name="fraud_detection_results.csv",
                            mime="text/csv"
                        )
            
            except Exception as e:
                st.error(f"❌ Lỗi khi xử lý file: {str(e)}")

# ============================================================================
# RUN APP
# ============================================================================
if __name__ == "__main__":
    main()