# US Treasury Curve Forecasting Project

Advanced End-to-End Machine Learning Pipeline for Forecasting the US Treasury Yield Curve using Multiple Modeling Approaches

## 📊 Project Overview

This project implements a comprehensive forecasting system for the US Treasury yield curve using various advanced modeling techniques. The goal is to predict future movements in Treasury yields across different maturities, with a particular focus on the yield spread between 10-year and 2-year Treasury rates.

### Key Features
- **Multi-Model Approach**: Baseline models, autoregression models, machine learning, and deep learning
- **Comprehensive Feature Engineering**: Advanced time series feature creation with lead-lag relationships
- **Statistical Analysis**: Stationarity tests, correlation analysis, and cross-correlation studies
- **Probabilistic Forecasting**: Monte Carlo simulations and uncertainty quantification
- **Performance Evaluation**: Multiple metrics and visualization tools

## 🏗️ Project Structure

```
Forecast_Treasury_Curve/
├── Dataset/
│   └── final_feature_library_all_features.csv    # Comprehensive feature dataset
├── notebooks/
│   ├── Step1_Preliminary_analysis.ipynb          # Data exploration and stationarity tests
│   ├── Step2_baseline_models.ipynb               # Baseline forecasting models
│   ├── Step3_Autoregression_model.ipynb          # ARIMA and autoregressive models
│   ├── Step4_ML_forecasting.ipynb                # Machine learning models (LightGBM, etc.)
│   └── Step5_Prophet_model.ipynb                 # Facebook Prophet implementation
└── README.md
```

## 📈 Methodology

### 1. Data Preprocessing & Analysis
- **Feature Engineering**: Creation of lag/lead features, rolling statistics, momentum indicators
- **Stationarity Testing**: ADF and KPSS tests for time series analysis
- **Correlation Analysis**: Pearson and Spearman correlations with significance testing
- **Cross-Correlation**: Optimal lead-lag relationship identification

### 2. Modeling Approaches

#### Baseline Models
- Simple moving averages
- Linear regression models
- Naive forecasting methods

#### Autoregression Models
- ARIMA (AutoRegressive Integrated Moving Average)
- VAR (Vector Autoregression)
- Seasonal decomposition

#### Machine Learning Models
- **LightGBM**: Gradient boosting with advanced feature engineering
- **Random Forest**: Ensemble methods for robust predictions
- **Support Vector Regression**: Non-linear relationship modeling
- **Neural Networks**: Deep learning approaches

#### Advanced Techniques
- **Prophet**: Facebook's time series forecasting tool
- **Monte Carlo Simulations**: Probabilistic forecasting
- **Bayesian Neural Networks**: Uncertainty quantification

### 3. Feature Engineering Strategy
```python
# Key features created:
- Lag features (1-5 periods)
- Rolling statistics (mean, std, min, max)
- Differencing features (1st and 2nd order)
- Percentage changes
- Momentum indicators
- Cross-correlation optimized features
```

## 🚀 Getting Started

### Prerequisites
```bash
# Core data science libraries
pip install pandas numpy matplotlib seaborn
pip install scikit-learn scipy statsmodels

# Time series specific
pip install prophet lightgbm

# Additional dependencies
pip install jupyter notebook
```

### Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/Forecast_Treasury_Curve.git
cd Forecast_Treasury_Curve
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

## 📋 Usage

### Step-by-Step Execution

1. **Data Exploration** (`Step1_Preliminary_analysis.ipynb`)
   - Load and examine the dataset
   - Perform stationarity tests
   - Initial data visualization

2. **Baseline Modeling** (`Step2_baseline_models.ipynb`)
   - Implement simple forecasting models
   - Establish performance benchmarks

3. **Autoregression Analysis** (`Step3_Autoregression_model.ipynb`)
   - ARIMA model fitting and validation
   - Parameter optimization

4. **Machine Learning** (`Step4_ML_forecasting.ipynb`)
   - Feature engineering pipeline
   - Model training and evaluation
   - Performance comparison

5. **Prophet Implementation** (`Step5_Prophet_model.ipynb`)
   - Facebook Prophet model
   - Trend and seasonality analysis

### Key Functions

#### Feature Engineering
```python
def create_lead_lag_features(df, target_col='Spread', max_lags=5):
    """
    Create comprehensive time series features including:
    - Lag features (1-5 periods)
    - Rolling statistics (3, 5, 10 periods)
    - Differencing and momentum features
    """
```

#### Correlation Analysis
```python
def correlation_analysis(df, target_col='Spread', threshold=0.1):
    """
    Perform comprehensive correlation analysis with:
    - Pearson and Spearman correlations
    - Statistical significance testing
    - Feature ranking by importance
    """
```

## 📊 Dataset Description

The project uses a comprehensive feature library containing:
- **Treasury Yields**: Various maturities (2Y, 5Y, 10Y, 30Y)
- **Economic Indicators**: GDP, inflation, employment data
- **Market Indicators**: VIX, equity indices, commodity prices
- **Technical Features**: Moving averages, momentum indicators
- **Derived Features**: Yield spreads, curvature measures

### Data Features
- **Time Period**: Historical daily data
- **Features**: 100+ engineered features
- **Target Variable**: 10Y-2Y Treasury spread (forward-looking)

## 📈 Results & Performance

### Model Performance Metrics
- **Mean Absolute Error (MAE)**
- **Root Mean Square Error (RMSE)**
- **R-squared (R²)**
- **Directional Accuracy**

### Key Findings
- Advanced feature engineering significantly improves model performance
- Ensemble methods (LightGBM) show superior results
- Cross-correlation analysis reveals optimal lead-lag relationships
- Probabilistic forecasting provides valuable uncertainty estimates

## 🔧 Technical Details

### Model Architecture
- **Feature Selection**: Correlation-based and mutual information
- **Hyperparameter Tuning**: Grid search and cross-validation
- **Ensemble Methods**: Model stacking and voting
- **Validation Strategy**: Time series cross-validation

### Performance Optimization
- **Memory Management**: Efficient data handling for large datasets
- **Computational Efficiency**: Vectorized operations and parallel processing
- **Model Persistence**: Save/load trained models for deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- US Treasury Department for yield data
- Federal Reserve Economic Data (FRED)
- Financial research community
- Open-source contributors

## 📞 Contact

For questions or collaboration opportunities:
- **Email**: [yash.sethi@mail.mcgill.ca]
- **LinkedIn**: [https://www.linkedin.com/in/yash-sethi24/]
- **GitHub**: [https://github.com/Yashsethi24]

---

**Note**: This project is for educational and research purposes. Financial forecasting involves inherent uncertainty and should not be used as the sole basis for investment decisions.
