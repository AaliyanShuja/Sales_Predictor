import streamlit as st
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from streamlit.components.v1 import html
from PIL import Image
import base64
import time
import requests
import io
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Page Config
st.set_page_config(
    page_title="AI Sales Forecaster Pro",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# Enhanced CSS with more animations and styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(30px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideInLeft {
        0% { opacity: 0; transform: translateX(-50px); }
        100% { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes slideInRight {
        0% { opacity: 0; transform: translateX(50px); }
        100% { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.7); }
        70% { transform: scale(1.05); box-shadow: 0 0 0 10px rgba(59, 130, 246, 0); }
        100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(59, 130, 246, 0); }
    }
    
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
        100% { transform: translateY(0px); }
    }
    
    @keyframes glow {
        0% { box-shadow: 0 0 5px rgba(16, 185, 129, 0.5); }
        50% { box-shadow: 0 0 20px rgba(16, 185, 129, 0.8), 0 0 30px rgba(16, 185, 129, 0.6); }
        100% { box-shadow: 0 0 5px rgba(16, 185, 129, 0.5); }
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        animation: fadeIn 1s ease-out;
    }
    
    .insight-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 1rem 0;
        animation: slideInLeft 0.8s ease-out;
        border-left: 5px solid #3B82F6;
        transition: all 0.3s ease;
    }
    
    .insight-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #10B981 0%, #34D399 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        animation: pulse 2s infinite;
        margin: 1rem 0;
    }
    
    .feature-button {
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 10px;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        animation: float 3s ease-in-out infinite;
        margin: 0.5rem;
    }
    
    .feature-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(59, 130, 246, 0.3);
    }
    
    .dashboard-section {
        background: #f8fafc;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        animation: slideInRight 0.8s ease-out;
    }
    
    .animated-chart {
        animation: glow 2s ease-in-out infinite alternate;
        border-radius: 10px;
        padding: 1rem;
        background: white;
    }
    
    .hero-container {
        display: flex;
        min-height: 60vh;
        margin-bottom: 2rem;
    }
    
    .hero-left {
        flex: 1;
        padding: 3rem;
        display: flex;
        flex-direction: column;
        justify-content: center;
        background: linear-gradient(135deg, #1e3a8a 0%, #1d4ed8 100%);
        color: white;
        border-radius: 20px 0 0 20px;
    }
    
    .hero-right {
        flex: 1;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        position: relative;
        overflow: hidden;
        border-radius: 0 20px 20px 0;
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .correlation-matrix {
        animation: fadeIn 1.5s ease-out;
    }
    
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid #f3f3f3;
        border-top: 3px solid #3498db;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .feature-showcase {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #10B981;
    }
    
    .welcome-section {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Advanced animated visualization components


def create_animated_chart_js():
    return """
    <div id="animated-chart-container" style="width: 100%; height: 400px; position: relative;">
        <canvas id="revenueChart" style="width: 100%; height: 100%;"></canvas>
    </div>
    
    <script>
    const canvas = document.getElementById('revenueChart');
    const ctx = canvas.getContext('2d');
    
    // Set canvas size
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    
    let animationId;
    let data = [];
    let time = 0;
    
    // Generate sample data
    for (let i = 0; i < 50; i++) {
        data.push({
            x: i,
            y: Math.sin(i * 0.1) * 50 + Math.random() * 30 + 100,
            revenue: Math.random() * 1000 + 500
        });
    }
    
    function drawChart() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Draw background gradient
        const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
        gradient.addColorStop(0, 'rgba(59, 130, 246, 0.1)');
        gradient.addColorStop(1, 'rgba(16, 185, 129, 0.1)');
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Draw grid
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
        ctx.lineWidth = 1;
        
        for (let i = 0; i < 10; i++) {
            const y = (canvas.height / 10) * i;
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(canvas.width, y);
            ctx.stroke();
        }
        
        // Draw animated line
        ctx.strokeStyle = '#10B981';
        ctx.lineWidth = 3;
        ctx.beginPath();
        
        const stepX = canvas.width / data.length;
        
        for (let i = 0; i < data.length; i++) {
            const x = i * stepX;
            const y = canvas.height - (data[i].y / 200 * canvas.height);
            
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.stroke();
        
        // Draw animated points
        data.forEach((point, index) => {
            const x = index * stepX;
            const y = canvas.height - (point.y / 200 * canvas.height);
            
            ctx.fillStyle = '#3B82F6';
            ctx.beginPath();
            ctx.arc(x, y, 4 + Math.sin(time + index * 0.1) * 2, 0, Math.PI * 2);
            ctx.fill();
        });
        
        // Update data for animation
        data.forEach((point, index) => {
            point.y += Math.sin(time + index * 0.1) * 0.5;
            point.y = Math.max(50, Math.min(150, point.y));
        });
        
        time += 0.05;
        animationId = requestAnimationFrame(drawChart);
    }
    
    drawChart();
    </script>
    """


def create_3d_animation():
    return """
    <div id="3d-container" style="width: 100%; height: 300px; background: linear-gradient(45deg, #667eea, #764ba2); border-radius: 15px; overflow: hidden;">
        <canvas id="3dChart" style="width: 100%; height: 100%;"></canvas>
    </div>
    
    <script>
    const canvas3d = document.getElementById('3dChart');
    const ctx3d = canvas3d.getContext('2d');
    
    canvas3d.width = canvas3d.offsetWidth;
    canvas3d.height = canvas3d.offsetHeight;
    
    let rotation = 0;
    let cubes = [];
    
    // Initialize cubes
    for (let i = 0; i < 20; i++) {
        cubes.push({
            x: Math.random() * canvas3d.width,
            y: Math.random() * canvas3d.height,
            z: Math.random() * 100,
            size: Math.random() * 20 + 10,
            speed: Math.random() * 2 + 1
        });
    }
    
    function draw3D() {
        ctx3d.clearRect(0, 0, canvas3d.width, canvas3d.height);
        
        // Sort cubes by z-index
        cubes.sort((a, b) => b.z - a.z);
        
        cubes.forEach(cube => {
            const scale = cube.z / 100;
            const alpha = scale;
            
            ctx3d.save();
            ctx3d.translate(cube.x, cube.y);
            ctx3d.rotate(rotation);
            
            ctx3d.fillStyle = `rgba(255, 255, 255, ${alpha * 0.8})`;
            ctx3d.fillRect(-cube.size * scale / 2, -cube.size * scale / 2, 
                          cube.size * scale, cube.size * scale);
            
            ctx3d.restore();
            
            // Update position
            cube.z -= cube.speed;
            if (cube.z <= 0) {
                cube.z = 100;
                cube.x = Math.random() * canvas3d.width;
                cube.y = Math.random() * canvas3d.height;
            }
        });
        
        rotation += 0.02;
        requestAnimationFrame(draw3D);
    }
    
    draw3D();
    </script>
    """


def create_particle_animation():
    return """
    <div id="particle-container" style="width: 100%; height: 250px; background: #1a202c; border-radius: 15px; overflow: hidden; position: relative;">
        <canvas id="particleChart" style="width: 100%; height: 100%;"></canvas>
    </div>
    
    <script>
    const canvasP = document.getElementById('particleChart');
    const ctxP = canvasP.getContext('2d');
    
    canvasP.width = canvasP.offsetWidth;
    canvasP.height = canvasP.offsetHeight;
    
    let particles = [];
    
    // Create particles
    for (let i = 0; i < 100; i++) {
        particles.push({
            x: Math.random() * canvasP.width,
            y: Math.random() * canvasP.height,
            vx: (Math.random() - 0.5) * 2,
            vy: (Math.random() - 0.5) * 2,
            size: Math.random() * 3 + 1,
            color: `hsl(${Math.random() * 60 + 200}, 100%, 70%)`
        });
    }
    https://salespredictor-production.up.railway.app/upload_csv
    function drawParticles() {
        ctxP.fillStyle = 'rgba(26, 32, 44, 0.1)';
        ctxP.fillRect(0, 0, canvasP.width, canvasP.height);
        
        particles.forEach(particle => {
            ctxP.fillStyle = particle.color;
            ctxP.beginPath();
            ctxP.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
            ctxP.fill();
            
            // Update position
            particle.x += particle.vx;
            particle.y += particle.vy;
            
            // Bounce off edges
            if (particle.x <= 0 || particle.x >= canvasP.width) particle.vx *= -1;
            if (particle.y <= 0 || particle.y >= canvasP.height) particle.vy *= -1;
        });
        
        requestAnimationFrame(drawParticles);
    }
    
    drawParticles();
    </script>
    """


# Sidebar configuration
with st.sidebar:
    st.markdown("## 🎛️ Dashboard Controls")

    # Analysis options
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Revenue Insights", "Feature Correlation",
            "Trend Analysis", "Performance Metrics"]
    )

    # Visualization options
    st.markdown("### 📊 Visualization Options")
    show_animations = st.checkbox("Enable Animations", value=True)
    chart_theme = st.selectbox("Chart Theme", ["Default", "Dark", "Colorful"])

    # Advanced filters
    st.markdown("### 🔍 Advanced Filters")
    date_range = st.date_input("Date Range", value=[])
    revenue_threshold = st.slider("Revenue Threshold", 0, 10000, 1000)

# Main dashboard header
st.markdown("""
<div class="main-header">
    <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">🚀 AI Sales Forecaster Pro</h1>
    <p style="font-size: 1.2rem; opacity: 0.9;">Advanced Analytics & Predictive Insights Dashboard</p>
</div>
""", unsafe_allow_html=True)

# Hero section with enhanced animation
st.markdown("""
<div class="hero-container">
    <div class="hero-left">
        <h2 style="font-size: 2.5rem; margin-bottom: 1rem;">Transform Your Sales Data</h2>
        <p style="font-size: 1.1rem; line-height: 1.6; margin-bottom: 2rem;">
            Leverage cutting-edge machine learning algorithms to predict sales trends, 
            analyze customer behavior, and optimize your revenue streams with unprecedented accuracy.
        </p>
        <div style="display: flex; gap: 1rem;">
            <button class="feature-button" onclick="window.scrollTo(0, document.querySelector('.stats-grid').offsetTop)">📈 Predict Revenue</button>
            <button class="feature-button" onclick="window.scrollTo(0, document.querySelector('.dashboard-section').offsetTop)">🔍 Analyze Trends</button>
            <button class="feature-button" onclick="window.scrollTo(0, document.querySelector('.insight-card').offsetTop)">💡 Get Insights</button>
        </div>
    </div>
    <div class="hero-right">
""", unsafe_allow_html=True)

# Animated chart in hero section
if show_animations:
    html(create_animated_chart_js(), height=400)

st.markdown("""
    </div>
</div>
""", unsafe_allow_html=True)

# Three animations section
if show_animations:
    st.markdown("## 🎨 Interactive Animations")

    anim_col1, anim_col2, anim_col3 = st.columns(3)

    with anim_col1:
        st.markdown("### 📊 Revenue Flow")
        html(create_animated_chart_js(), height=250)

    with anim_col2:
        st.markdown("### 🎲 3D Data Cubes")
        html(create_3d_animation(), height=250)

    with anim_col3:
        st.markdown("### ✨ Data Particles")
        html(create_particle_animation(), height=250)

# File upload section
uploaded_file = st.file_uploader("📂 Upload Your Sales Data (CSV)", type="csv")

if uploaded_file is not None:
    try:
        # Load and display data
        df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode("utf-8")))

        # Check for required columns
        required_columns = ['order_id', 'order_date', 'sku', 'color', 'size', 'unit_price',
                            'quantity', 'revenue', 'age', 'discount', 'customer_rating',
                            'stock', 'category', 'category_id', 'category_avg_price',
                            'category_total_revenue', 'category_popularity', 'holiday_type']

        missing_columns = [
            col for col in required_columns if col not in df.columns]

        if missing_columns:
            st.warning(
                f"⚠️ The dataset is missing some required columns: {', '.join(missing_columns)}")
            st.info("For best results, please ensure your dataset contains all these columns: " +
                    ", ".join(required_columns))

        # Data overview section
        st.markdown("## 📋 Data Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Records</h3>
                <h1>{len(df):,}</h1>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #F59E0B 0%, #F97316 100%);">
                <h3>Features</h3>
                <h1>{len(df.columns)}</h1>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #8B5CF6 0%, #A855F7 100%);">
                <h3>Numeric Columns</h3>
                <h1>{len(numeric_cols)}</h1>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            missing_data = df.isnull().sum().sum()
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #EF4444 0%, #F87171 100%);">
                <h3>Missing Values</h3>
                <h1>{missing_data}</h1>
            </div>
            """, unsafe_allow_html=True)

        # Interactive data exploration
        st.markdown("### 🔍 Interactive Data Exploration")

        # Tabs for different analysis types
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📊 Revenue Analysis", "🔗 Feature Correlation", "📈 Trend Analysis", "🎯 Predictions"])

        with tab1:
            st.markdown('<div class="dashboard-section">',
                        unsafe_allow_html=True)

            if analysis_type == "Revenue Insights" or analysis_type == "Revenue Analysis":
                revenue_cols = [col for col in df.columns if 'revenue' in col.lower(
                ) or 'sales' in col.lower() or 'amount' in col.lower()]

                if revenue_cols:
                    selected_revenue_col = st.selectbox(
                        "Select Revenue Column", revenue_cols)

                    # Revenue distribution
                    fig = px.histogram(df, x=selected_revenue_col,
                                       title=f"Revenue Distribution - {selected_revenue_col}",
                                       color_discrete_sequence=['#3B82F6'])
                    fig.update_layout(template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)

                    # 3D Scatter Plot
                    st.markdown("### 3D Revenue Analysis")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        x_axis = st.selectbox("X-Axis", numeric_cols, index=0)
                    with col2:
                        y_axis = st.selectbox("Y-Axis", numeric_cols, index=1)
                    with col3:
                        z_axis = st.selectbox("Z-Axis", numeric_cols, index=2)

                    fig_3d = px.scatter_3d(df, x=x_axis, y=y_axis, z=z_axis,
                                           color=selected_revenue_col,
                                           title=f"3D Revenue Analysis: {x_axis} vs {y_axis} vs {z_axis}",
                                           color_continuous_scale='Viridis')
                    st.plotly_chart(fig_3d, use_container_width=True)

                    # Top revenue insights
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("#### 💰 Revenue Insights")
                        total_revenue = df[selected_revenue_col].sum()
                        avg_revenue = df[selected_revenue_col].mean()
                        max_revenue = df[selected_revenue_col].max()

                        st.markdown(f"""
                        <div class="insight-card">
                            <h4>Key Metrics</h4>
                            <p><strong>Total Revenue:</strong> ${total_revenue:,.2f}</p>
                            <p><strong>Average Revenue:</strong> ${avg_revenue:.2f}</p>
                            <p><strong>Maximum Revenue:</strong> ${max_revenue:.2f}</p>
                            <p><strong>Revenue Range:</strong> ${df[selected_revenue_col].min():.2f} - ${max_revenue:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        # Revenue by category
                        if 'category' in df.columns:
                            revenue_by_cat = df.groupby(
                                'category')[selected_revenue_col].sum().sort_values(ascending=False)

                            fig = px.bar(x=revenue_by_cat.index, y=revenue_by_cat.values,
                                         title=f"Revenue by Category",
                                         color=revenue_by_cat.values,
                                         color_continuous_scale='Viridis')
                            st.plotly_chart(fig, use_container_width=True)

                else:
                    st.warning("No revenue columns detected in the dataset.")

            st.markdown('</div>', unsafe_allow_html=True)

        with tab2:
            st.markdown(
                '<div class="dashboard-section correlation-matrix">', unsafe_allow_html=True)

            numeric_cols = df.select_dtypes(include=[np.number]).columns

            if len(numeric_cols) > 1:
                st.markdown("#### 🔗 Feature Correlation Analysis")

                # Correlation matrix
                corr_data = df[numeric_cols].corr()

                fig = px.imshow(corr_data,
                                title="Feature Correlation Matrix",
                                color_continuous_scale='RdBu',
                                aspect="auto")
                fig.update_layout(template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

                # Feature relationships
                st.markdown("#### 📈 Feature Relationships")
                col1, col2 = st.columns(2)

                with col1:
                    feature1 = st.selectbox(
                        "Select Feature 1", numeric_cols, key="feat1")

                with col2:
                    feature2 = st.selectbox(
                        "Select Feature 2", numeric_cols, key="feat2")

                if feature1 != feature2:
                    fig = px.scatter(df, x=feature1, y=feature2,
                                     title=f"Relationship: {feature1} vs {feature2}",
                                     trendline="ols",
                                     color_discrete_sequence=['#10B981'])
                    st.plotly_chart(fig, use_container_width=True)

                    # Calculate correlation coefficient
                    correlation = df[feature1].corr(df[feature2])
                    st.markdown(f"""
                    <div class="insight-card">
                        <h4>Correlation Insight</h4>
                        <p><strong>Correlation Coefficient:</strong> {correlation:.3f}</p>
                        <p><strong>Relationship Strength:</strong> 
                        {'Strong' if abs(correlation) > 0.7 else 'Moderate' if abs(correlation) > 0.3 else 'Weak'}</p>
                        <p><strong>Direction:</strong> {'Positive' if correlation > 0 else 'Negative'}</p>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        with tab3:
            st.markdown('<div class="dashboard-section">',
                        unsafe_allow_html=True)

            # Time series analysis
            if 'order_date' in df.columns:
                st.markdown("#### 📈 Time Series Trend Analysis")

                try:
                    df['order_date'] = pd.to_datetime(df['order_date'])

                    if 'revenue' in df.columns:
                        # Time series plot
                        df_sorted = df.sort_values('order_date')

                        fig = px.line(df_sorted, x='order_date', y='revenue',
                                      title="Revenue Trend Over Time",
                                      color_discrete_sequence=['#3B82F6'])
                        fig.update_layout(template="plotly_white")
                        st.plotly_chart(fig, use_container_width=True)

                        # Trend analysis
                        monthly_revenue = df_sorted.groupby(
                            df_sorted['order_date'].dt.to_period('M'))['revenue'].sum()

                        fig = px.bar(x=monthly_revenue.index.astype(str), y=monthly_revenue.values,
                                     title="Monthly Revenue Trend",
                                     color=monthly_revenue.values,
                                     color_continuous_scale='Viridis')
                        st.plotly_chart(fig, use_container_width=True)

                        # Interactive time series decomposition
                        st.markdown("#### 🕰️ Time Series Decomposition")

                        if st.button("Analyze Time Series Components"):
                            from statsmodels.tsa.seasonal import seasonal_decompose

                            # Resample to daily data
                            daily_revenue = df_sorted.set_index(
                                'order_date')['revenue'].resample('D').sum()

                            # Fill any missing values
                            daily_revenue = daily_revenue.fillna(
                                daily_revenue.rolling(7, min_periods=1).mean())

                            # Decompose the time series
                            result = seasonal_decompose(
                                daily_revenue, model='additive', period=30)

                            # Plot decomposition
                            fig = make_subplots(
                                rows=4, cols=1, shared_xaxes=True)

                            fig.add_trace(go.Scatter(
                                x=daily_revenue.index, y=daily_revenue, name='Observed'), row=1, col=1)
                            fig.add_trace(go.Scatter(
                                x=result.trend.index, y=result.trend, name='Trend'), row=2, col=1)
                            fig.add_trace(go.Scatter(
                                x=result.seasonal.index, y=result.seasonal, name='Seasonal'), row=3, col=1)
                            fig.add_trace(go.Scatter(
                                x=result.resid.index, y=result.resid, name='Residual'), row=4, col=1)

                            fig.update_layout(
                                height=800, title_text="Time Series Decomposition")
                            st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Error processing date column: {e}")

            else:
                st.info("No date columns detected for trend analysis.")

            st.markdown('</div>', unsafe_allow_html=True)

        with tab4:
            st.markdown('<div class="dashboard-section">',
                        unsafe_allow_html=True)
            st.markdown("#### 🎯 AI‑Powered Predictions")

            if st.button("🔄 Generate Predictions", key="local_pred"):
                with st.spinner('Analyzing data and generating predictions...'):
                    try:
                        # Simulate prediction (in a real app, you'd use your actual model)
                        if 'revenue' in df.columns:
                            # Create some simulated predictions
                            np.random.seed(42)
                            df["Predicted_Revenue"] = df['revenue'] * \
                                (1 + np.random.normal(0, 0.1, len(df)))
                            df["Next_Month_Revenue"] = df['revenue'] * \
                                (1 + np.random.normal(0.05, 0.15, len(df)))

                            st.success("✅ Predictions generated successfully!")

                            # Show prediction results
                            st.markdown("### 📊 Prediction Results")

                            col1, col2 = st.columns(2)

                            with col1:
                                st.markdown("#### Actual vs Predicted Revenue")
                                fig = px.scatter(df, x='revenue', y='Predicted_Revenue',
                                                 trendline="ols",
                                                 title="Actual vs Predicted Revenue",
                                                 labels={'revenue': 'Actual Revenue', 'Predicted_Revenue': 'Predicted Revenue'})
                                st.plotly_chart(fig, use_container_width=True)

                            with col2:
                                st.markdown("#### Next Month Revenue Forecast")
                                fig = px.histogram(df, x='Next_Month_Revenue',
                                                   title="Distribution of Next Month Revenue Predictions",
                                                   nbins=30)
                                st.plotly_chart(fig, use_container_width=True)

                            # Show top predictions
                            st.markdown("### 🏆 Top Predictions")
                            top_predictions = df.nlargest(10, 'Next_Month_Revenue')[
                                ['order_id', 'category', 'revenue', 'Next_Month_Revenue']]
                            st.dataframe(top_predictions.style.format({
                                'revenue': '${:,.2f}',
                                'Next_Month_Revenue': '${:,.2f}'
                            }))

                            # Download predictions
                            csv = df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download Predictions",
                                data=csv,
                                file_name='sales_predictions.csv',
                                mime='text/csv'
                            )
                        else:
                            st.error(
                                "Revenue column not found - cannot generate predictions")
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
            else:
                st.markdown("""
                <div class="feature-showcase">
                    <h3>🔍 Predictive Analysis</h3>
                    <p>Click the button to generate revenue predictions using our AI model.</p>
                    <p><strong>Required Features:</strong> order_id, order_date, sku, color, size, unit_price, 
                    quantity, revenue, age, discount, customer_rating, stock, category, 
                    category_id, category_avg_price, category_total_revenue, 
                    category_popularity, holiday_type</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
