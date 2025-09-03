# =========================================================================
# app.py - The Final, Complete, and Verified Version
# =========================================================================

import pandas as pd
import numpy as np
from flask import Flask, request, render_template
import joblib
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import plotly.express as px
from matplotlib.colors import LinearSegmentedColormap
import colorsys

# --- 1. Initialize Flask App ---
app = Flask(__name__)

# --- 2. Load Global Assets ---
try:
    solution_path = os.path.join('models', 'cost_optimized_churn_solution_v3.joblib')
    loaded_solution = joblib.load(solution_path)
    MODEL_PIPELINE = loaded_solution['model_pipeline']
    OPTIMAL_THRESHOLD = loaded_solution['optimal_threshold']
    print("✅ Cost-optimized solution (v3.0) loaded.")
    
    df_for_viz = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df_for_viz['TotalCharges'] = pd.to_numeric(df_for_viz['TotalCharges'], errors='coerce').fillna(0)
    df_for_viz.drop('customerID', axis=1, inplace=True)
    df_for_viz['SeniorCitizen'] = df_for_viz['SeniorCitizen'].apply(lambda x: 'Yes' if x == 1 else 'No')
    cols_to_clean = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines']
    for col in cols_to_clean:
        df_for_viz[col] = df_for_viz[col].replace(['No internet service', 'No phone service'], 'No')
    DF_CLEAN = df_for_viz
    print("✅ Clean data for visualizations loaded.")

except Exception as e:
    print(f"❌ FATAL ERROR loading assets: {e}")
    MODEL_PIPELINE = None
    DF_CLEAN = None

# --- 3. Helper Functions ---
def safe_float(val, default=0.0):
    try: return float(val)
    except (ValueError, TypeError): return default

def safe_int(val, default=0):
    try: return int(float(val))
    except (ValueError, TypeError): return default

# --- 4. Flask Routes ---
@app.route('/about')
def about():
    """Renders the About Us page."""
    return render_template('aboutus.html')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not MODEL_PIPELINE: return "Server Error: Model not loaded.", 500
    form_data = request.form.to_dict()
    data_for_df = {}
    
    categorical_features = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    for col in categorical_features:
        data_for_df[col] = [str(form_data.get(col, 'No'))]

    data_for_df['tenure'] = [safe_int(form_data.get('tenure'))]
    data_for_df['MonthlyCharges'] = [safe_float(form_data.get('MonthlyCharges'))]
    data_for_df['TotalCharges'] = [safe_float(form_data.get('TotalCharges'))]
    
    input_df = pd.DataFrame(data_for_df)
    
    churn_probability = MODEL_PIPELINE.predict_proba(input_df)[0][1]
    final_prediction = 1 if churn_probability >= OPTIMAL_THRESHOLD else 0

    explanation, suggestion = generate_explanation_and_suggestion(input_df)
    
    if final_prediction == 1:
        output_text, color = "CUSTOMER IS HIGH RISK (LIKELY TO CHURN)", "#ff4757"
    else:
        output_text, color = "Customer is Low Risk (Likely to Stay)", "#2ed573"
        
    output_confidence = f"{churn_probability * 100:.2f}%"

    return render_template('result_detailed.html', 
                           prediction_text=output_text, confidence_text=output_confidence,
                           result_color=color, explanation=explanation, suggestion=suggestion)

@app.route('/explore')
def explore():
    if DF_CLEAN is None: return "Server Error: Visualization data not loaded.", 500
    
    # THE FIX IS HERE: Using 'number' string instead of pd.np.number
    categorical_cols = DF_CLEAN.select_dtypes(include=['object']).columns.drop('Churn').tolist()
    numerical_cols = DF_CLEAN.select_dtypes(include='number').columns.tolist()
    
    return render_template('explore.html', 
                           categorical_cols=categorical_cols, numerical_cols=numerical_cols)

# Custom color palettes
CUSTOM_PALETTE = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEEAD", "#D4A5A5", "#9B9B9B"]
CHURN_COLORS = {"No": "#4ECDC4", "Yes": "#FF6B6B"}


@app.route('/generate_plot', methods=['GET'])
def generate_plot():
    if DF_CLEAN is None:
        return {"error": "Server Error: Visualization data not loaded."}, 500
    
    plot_type = request.args.get('type')
    x_col = request.args.get('x')
    y_col = request.args.get('y')

    # --- THEME AND SIZE STANDARDIZATION ---
    plt.style.use('dark_background')
    
    if plot_type == 'pie':
        fig, ax = plt.subplots(figsize=(6, 6)) # Square size for pie charts
    else:
        fig, ax = plt.subplots(figsize=(9, 5.5)) # Wider size for other plots
    
    bg_color = '#0D1117'
    grid_color = '#30363d'
    text_color = '#c9d1d9'
    accent_color_blue = '#3B82F6' # A professional blue for single-color plots
    
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    try:
        # --- PLOTTING LOGIC WITH NEW COLOR PALETTES ---
        
        if plot_type == 'univariate_bar':
            sns.countplot(data=DF_CLEAN, x=x_col, ax=ax, color=accent_color_blue)
            analysis_text = f"Distribution of customers by '{x_col}'."
            
        elif plot_type == 'bivariate_bar_churn':
            # NEW: Using a clean Blue and Red palette for churn
            churn_palette = {'No': '#3B82F6', 'Yes': '#EF4444'}
            sns.countplot(data=DF_CLEAN, x=x_col, hue='Churn', ax=ax, palette=churn_palette)
            ax.legend(title='Churn Status')
            analysis_text = f"Comparison of churn status across '{x_col}'."
            
        elif plot_type == 'box' and y_col:
            sns.boxplot(data=DF_CLEAN, x=x_col, y=y_col, ax=ax, color=accent_color_blue)
            analysis_text = f"Box plot showing the distribution of '{y_col}' for each '{x_col}' category."
            
        elif plot_type == 'pie':
            # NEW: Using your requested Orange and Blue palette
            pie_colors = ['#FFA500', '#3B82F6', '#22C55E', '#8B5CF6', '#F97316'] # Orange, Blue, Green, etc. for more categories
            ax.set_aspect('equal')
            counts = DF_CLEAN[x_col].value_counts()
            labels = counts.index
            
            ax.pie(counts, 
                   labels=labels,
                   colors=pie_colors,
                   autopct='%1.1f%%',
                   textprops={'color': text_color, 'fontsize': 11, 'weight': 'bold'},
                   pctdistance=0.7)
            
            analysis_text = f"Percentage breakdown of customers by '{x_col}'."

        # --- STYLING LOGIC (Applied to all plots except pie) ---
        if plot_type != 'pie':
            ax.grid(True, which='major', axis='y', color=grid_color, alpha=0.5, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color(grid_color)
            ax.spines['left'].set_color(grid_color)
            ax.tick_params(colors=text_color, bottom=False, left=False)
            plt.xticks(rotation=20, ha='right')
        
        ax.set_title(f'Analysis of {x_col.title()}', color=text_color, pad=20, fontsize=16)
        
        # --- SAVE AND RETURN IMAGE ---
        plt.tight_layout(pad=1.5)
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', dpi=120, facecolor=bg_color)
        img.seek(0)
        plt.close(fig)

        return {
            'image_data': base64.b64encode(img.read()).decode('utf-8'),
            'analysis_text': analysis_text
        }

    except Exception as e:
        print(f"Error generating plot: {e}")
        return {"error": f"Error generating plot: {str(e)}"}, 500
    

# --- 5. Utility Functions ---
def generate_explanation_and_suggestion(customer_df):
    top_risk_factors, suggestion_text = [], "Review customer account for satisfaction."
    if customer_df['Contract'].iloc[0] == 'Month-to-month':
        top_risk_factors.append("Being on a Month-to-month contract is a primary churn risk factor.")
        suggestion_text = "Priority : Offer an incentive to upgrade to a 1- or 2-year contract."
    if customer_df['InternetService'].iloc[0] == 'Fiber optic':
        top_risk_factors.append("Having Fiber Optic internet is associated with higher churn.")
        if customer_df['TechSupport'].iloc[0] == 'No' and suggestion_text.startswith("Review"):
            suggestion_text = "Priority : Offer free/discounted Tech Support to this high-risk Fiber customer."
    if customer_df['tenure'].iloc[0] < 12:
        top_risk_factors.append(f"A **low tenure** ({customer_df['tenure'].iloc[0]} months) indicates a higher flight risk.")
    if not top_risk_factors:
        top_risk_factors.append("This customer does not exhibit the most common high-risk factors.")
    return top_risk_factors, suggestion_text

# --- 6. Run the Application ---
if __name__ == "__main__":
    app.run(debug=True)