# 📊 Sales Prediction Web App (Streamlit + LightGBM)

This is a web-based Sales Prediction project built using **Streamlit** and **LightGBM**. The application takes in historical sales data, applies preprocessing, and uses a trained LightGBM model to predict future sales trends.

---

## 🚀 Live Demo
If deployed on Streamlit Cloud, you can access it here:  
👉 [Live App Link](https://your-app-name.streamlit.app)

---

## 📌 Features

- 📁 Upload historical sales data (CSV)
- 🧼 Preprocesses raw data automatically
- 📈 Visualizes actual vs predicted sales using a signal graph
- 🧠 Uses a LightGBM model trained on past trends
- 💾 Download prediction results as CSV
- 💡 Intuitive UI with custom design (royal blue theme and sky-blue square design elements)

---

## 📂 Dataset Columns

The model expects a dataset with the following columns:

| Column Name  | Description                         |
|--------------|-------------------------------------|
| `order_date` | Date of the order                   |
| `SKU`        | Stock Keeping Unit (product code)   |
| `color`      | Color of the item                   |
| `size`       | Size of the item                    |
| `unit_price` | Price per item                      |
| `quantity`   | Number of units sold                |
| `revenue`    | Total revenue from the sale         |

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Backend Model:** LightGBM (trained locally)
- **Languages:** Python
- **Libraries Used:**  
  - `pandas`, `numpy`, `matplotlib`, `seaborn`  
  - `lightgbm` for model training  
  - `scikit-learn` for preprocessing  
  - `joblib` for saving/loading the model  

---

## 📦 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/AaliyanShuja/sales_predictor.git
cd sales_predictor

# Install required packages
pip install -r requirements.txt

# Run the Streamlit app
streamlit run main.py
