import streamlit as st
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
from   streamlit.components.v1 import html
from   PIL import Image
import base64
import time
import requests
import io

# Page Config
st.set_page_config(
    page_title="AI Sales Forecaster",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for pop-up and animations
st.markdown("""
<style>
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    .popup {
        animation: fadeIn 0.8s ease-out;
        background: white;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        padding: 2rem;
        margin: 2rem 0;
    }
    
    .hero-container {
        display: flex;
        min-height: 100vh;
    }
    
    .hero-left {
        flex: 1;
        padding: 4rem;
        display: flex;
        flex-direction: column;
        justify-content: center;
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
    }
    
    .hero-right {
        flex: 1;
        background: url('https://images.unsplash.com/photo-1620712943543-bcc4688e7485?ixlib=rb-4.0.3&auto=format&fit=crop&w=1080&q=80') center/cover no-repeat;
        position: relative;
        overflow: hidden;
    }
    
    .graph-animation {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(30, 58, 138, 0.7);
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .ml-text {
        font-size: 1.1rem;
        line-height: 1.8;
        margin-top: 2rem;
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)

# Animated Graph JavaScript
graph_js = """
<script>
// Create canvas for animated graph
const canvas = document.createElement('canvas');
canvas.width = 800;
canvas.height = 400;
canvas.style.width = '100%';
canvas.style.height = '100%';
document.currentScript.parentElement.appendChild(canvas);

const ctx = canvas.getContext('2d');
const width = canvas.width;
const height = canvas.height;

// Generate random financial data
let data = [];
let currentX = 0;
const dataLength = 100;

for (let i = 0; i < dataLength; i++) {
    data.push(Math.random() * 100 + 50);
}

// Animation function
function animate() {
    ctx.clearRect(0, 0, width, height);
    
    // Draw grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
    ctx.lineWidth = 1;
    
    // Horizontal grid
    for (let i = 0; i < 5; i++) {
        ctx.beginPath();
        ctx.moveTo(0, height / 5 * i);
        ctx.lineTo(width, height / 5 * i);
        ctx.stroke();
    }
    
    // Draw line graph
    ctx.strokeStyle = '#10B981';
    ctx.lineWidth = 3;
    ctx.beginPath();
    
    const step = width / dataLength;
    for (let i = 0; i < dataLength; i++) {
        const x = i * step;
        const y = height - (data[i] / 150 * height);
        
        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    }
    ctx.stroke();
    
    // Add moving point
    ctx.fillStyle = 'white';
    ctx.beginPath();
    const currentY = height - (data[currentX % dataLength] / 150 * height);
    ctx.arc(currentX % dataLength * step, currentY, 5, 0, Math.PI * 2);
    ctx.fill();
    
    // Update data for next frame (slight random walk)
    if (currentX % 3 === 0) {
        data = data.map(val => {
            const change = (Math.random() - 0.5) * 3;
            return Math.max(50, Math.min(150, val + change));
        });
    }
    https://salespredictor-production.up.railway.app/upload_csv
    currentX++;
    requestAnimationFrame(animate);
}

// Start animation
animate();
</script>
"""

# Hero Section
st.markdown("""
<div class="hero-container">
    <div class="hero-left">
        <h1 style="font-size: 3rem; margin-bottom: 1rem;">AI-Powered Sales Forecasting</h1>
        <div class="ml-text">
            Machine Learning (ML) is transforming the way businesses operate by delivering accurate predictions and high-level precision in decision-making. In a world driven by data, ML enables companies to analyze historical trends, forecast future outcomes, and make smarter, faster choices. Whether it's predicting sales trends, understanding customer behavior, or managing inventory, ML helps turn raw data into valuable insights. This not only improves efficiency and reduces human error but also empowers businesses to stay ahead of the competition with data-backed strategies. Embracing ML means unlocking new levels of growth, innovation, and customer satisfaction.
        </div>
    </div>
    <div class="hero-right">
        <div class="graph-animation">
""", unsafe_allow_html=True)

# Render the animated graph
html(graph_js, height=400)

st.markdown("""
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Dashboard Pop-up
st.markdown("""
<div class="popup">
    <h2 style="color: #1e3a8a; text-align: center;">📊 Sales Prediction Dashboard</h2>
    <p style="text-align: center;">Upload your data to generate AI-powered sales forecasts</p>
""", unsafe_allow_html=True)


# FastAPI endpoint
FASTAPI_URL = "https://salespredictor-production.up.railway.app/upload-csv" #/default/predict_batch_predict_batch_post  
st.title("📊 Sales Prediction Dashboard")

uploaded_file = st.file_uploader("📂 Upload CSV", type="csv")

if uploaded_file is not None:
    try:
        # Read and display uploaded CSV
        df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode("utf-8")))
        st.subheader("🔍 Uploaded Data")
        st.dataframe(df)

        # --- Predict using FastAPI ---
        if st.button("🔮 Predict with FastAPI"):
            with st.spinner("Sending file to FastAPI..."):
                response = requests.post(
                    FASTAPI_URL,
                    files={"file": uploaded_file.getvalue()}
                )

            if response.status_code == 200:
                result = response.json()
                df["Predicted Revenue"] = result["predictions"]
                st.success(f"✅ Predicted {result['rows']} records.")
                st.dataframe(df)
            else:
                st.error("❌ FastAPI request failed.")
                try:
                    st.json(response.json())
                except:
                    st.text(response.text)

        # --- Analyze Locally ---
        if st.button("🔄 Analyze Locally"):
            with st.spinner('Analyzing data locally...'):
                time.sleep(2)
                model = lgb.Booster(model_file="lightgbm_model.txt")
                model_features = model.feature_name()

                for col in model_features:
                    if col not in df.columns:
                        df[col] = 0  # Add missing columns as 0

                df["Predicted Revenue"] = model.predict(df[model_features])
                total_predicted = df["Predicted Revenue"].sum()
                avg_pred = df["Predicted Revenue"].mean()
                max_pred = df["Predicted Revenue"].max()

                st.success("✅ Local Analysis Complete")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div class="pulse" style="background: linear-gradient(135deg, #10B981 0%, #34D399 100%); 
                                padding: 2rem; border-radius: 15px; color: white; text-align: center;">
                        <h3>Total Predicted Revenue</h3>
                        <h1 style="margin: 0.5em 0;">${total_predicted:,.2f}</h1>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div style="background: #f8fafc; padding: 2rem; border-radius: 15px;">
                        <h4 style="color: #1e3a8a; margin-top: 0;">Key Insights</h4>
                        <p>• Predicted from {len(df)} transactions</p>
                        <p>• Average sale: ${avg_pred:.2f}</p>
                        <p>• Max predicted: ${max_pred:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Revenue trend
                if "order_date" in df.columns:
                    st.markdown("### 📈 Revenue Trend Forecast")
                    df["order_date"] = pd.to_datetime(df["order_date"])
                    df = df.sort_values("order_date")

                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(df["order_date"], df["Predicted Revenue"].cumsum(), 
                            color='#3B82F6', linewidth=3)
                    ax.fill_between(df["order_date"], df["Predicted Revenue"].cumsum(), 
                                    color='#3B82F6', alpha=0.1)
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Cumulative Revenue ($)")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)

        # --- Download Button ---
        st.download_button(
            "⬇️ Download Predictions as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="predicted_sales.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
