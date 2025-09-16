# Real Estate Market Insights with Predictive Analytics 🏘️📊

A real-world data science project that leverages **open data, machine learning, and business intelligence tools** to predict property prices and provide neighborhood-wise insights into Edmonton’s housing market.  

The project bridges a key gap: while Edmonton’s property data is publicly available, it’s often **not user-friendly** for buyers, sellers, or investors. Most decisions are still made through word of mouth. This project aims to **turn raw data into actionable insights** — helping people understand fair property values and market dynamics at the neighborhood level.  

## 🎯 Project Overview

This project analyzes **City of Edmonton’s Open Data** on residential properties to build a **predictive model for property valuation** and create **market analysis dashboards**.  

The pipeline includes:  
- Extracting and cleaning raw data (CSV format)  
- Performing **exploratory data analysis (EDA)** and correlation studies  
- Applying **machine learning models** for price prediction  
- Building a **Power BI dashboard** for neighborhood-wise insights  

### Key Research Questions
- What are the primary factors that influence housing prices in Edmonton?
- Which neighborhoods and property types offer the highest return on investment?
- Can we accurately predict property price categories using machine learning?
- How do seasonal trends affect real estate market dynamics?
- What are the emerging patterns in the Edmonton real estate market?

## 🚀 Project Highlights

- **End-to-End Pipeline**: From raw Edmonton Open Data to predictive insights  
- **Comprehensive EDA**: Normalization, outlier handling, feature correlation  
- **Machine Learning Models**: Random Forest (best accuracy), Gradient Boosting, Decision Tree, Linear/Ridge/Lasso Regression, SVM  
- **Visualization**: Power BI dashboard with interactive filters and neighborhood-level market analysis  
- **Actionable Outcomes**: Helps buyers, sellers, and analysts understand where and why property values differ across Edmonton  

## 📊 Key Findings & Insights

- **Price Prediction Accuracy**: Achieved 85%+ accuracy in property price category classification
- **Feature Importance**: Square footage, location, and listing priority are top price predictors
- **Market Trends**: Identified seasonal patterns and emerging neighborhood hotspots
- **Investment Opportunities**: Highlighted undervalued areas with high growth potential
- **Risk Assessment**: Developed risk metrics for different property types and locations

## 🛠️ Technical Stack

### Programming & Analytics
- **Python 3.8+**: Core programming language
- **Jupyter Notebook**: Interactive development environment
- **Pandas & NumPy**: Data manipulation and numerical computing
- **Scikit-learn**: Machine learning algorithms and model evaluation

### Visualization & Reporting
- **Power BI** → neighborhood-level dashboard and business intelligence 
- **Matplotlib & Seaborn**: Statistical visualizations
- **Plotly**: Interactive charts and geospatial mapping
- **Jupyter Widgets**: Interactive dashboard components

### Data Sources
- **City of Edmonton Open Data Portal**: Real estate transactions and property data
- **Geospatial APIs**: Location-based analytics and mapping

## 📂 Repository Structure

```
Real-Estate-Market-Insights-with-Predictive-Analytics/
├── 📓 Real_Estate_Analytics.ipynb                # Main analysis notebook with complete pipeline
├── 🐍 scrape.py                                  # Data extraction script from Edmonton API
├── 📊 real estate final.csv                      # Processed real estate dataset
├── 📋 README.md                                  # Project documentation
├── 📈 analysis_report.pdf                        # Detailed analysis report
├── 🎯 Real_Estate_Visualization.pbix             # Data Visualization using Power BI
└── 💳 Real_Estate_Visualization.png              # Snapshot of PowerBI Dashboard
```

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- Git

### Quick Start

1. **Clone the Repository**
   ```bash
   git clone https://github.com/17devraj/Real-Estate-Market-Insights-with-Predictive-Analytics.git
   cd Real-Estate-Market-Insights-with-Predictive-Analytics
   ```

2. **Create Virtual Environment** (Recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn plotly jupyter requests
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook Real_Estate_Analytics.ipynb
   ```

5. **Run Data Extraction** (Optional - data already included)
   ```bash
   python scrape.py
   ```

## 📈 Analysis Workflow

### 1. Data Acquisition & Preparation
- **API Integration**: Automated data extraction from City of Edmonton
- **Data Cleaning**: Handling missing values, outliers, and data type conversions
- **Feature Engineering**: Creating derived features and categorical encodings

### 2. Exploratory Data Analysis (EDA)
- **Descriptive Statistics**: Central tendencies and distributions
- **Correlation Analysis**: Identifying relationships between variables
- **Geospatial Analysis**: Location-based insights and mapping
- **Trend Analysis**: Temporal patterns and seasonal effects

### 3. Machine Learning Pipeline
- **Model Selection**: Comparison of multiple algorithms
- **Feature Selection**: Identifying most predictive variables
- **Model Training**: Cross-validation and hyperparameter tuning
- **Performance Evaluation**: Accuracy, precision, recall, and F1-score metrics

### 4. Business Intelligence & Reporting
- **Market Segmentation**: Property type and location-based analysis
- **ROI Calculations**: Investment return projections
- **Risk Assessment**: Market volatility and uncertainty quantification
- **Strategic Recommendations**: Data-driven investment guidance

## 🔍 Key Analytical Results

### Model Performance
- **Random Forest Classifier**: 87% accuracy in price category prediction
- **Logistic Regression**: 82% accuracy with excellent interpretability
- **Feature Importance**: Top 5 predictors identified and ranked

### Market Insights
- **High-Value Areas**: Premium neighborhoods with consistent appreciation
- **Emerging Markets**: Up-and-coming areas with growth potential
- **Investment Strategies**: Risk-adjusted return recommendations

### Statistical Findings
- **Price Correlation**: Strong positive correlation (r=0.78) between square footage and price
- **Location Premium**: Downtown properties command 25-30% price premium
- **Seasonal Trends**: Q2-Q3 showing highest market activity

## 📊 Visualizations Dashboard

![Image 1](Real_Estate_Visualization.png)

## 🎯 Business Applications

### For Real Estate Investors
- **Portfolio Optimization**: Data-driven property selection
- **Risk Management**: Market volatility assessment
- **Timing Analysis**: Optimal buy/sell timing recommendations

### For Real Estate Professionals
- **Market Analysis**: Comprehensive neighborhood reports
- **Pricing Strategy**: Competitive pricing recommendations
- **Client Advisory**: Evidence-based market guidance

### For Policy Makers
- **Market Monitoring**: Real-time market health indicators
- **Urban Planning**: Development opportunity identification
- **Economic Impact**: Housing market contribution to local economy

## 🔮 Future Enhancements

### Technical Improvements
- **Real-time Data Pipeline**: Live API integration for current market data
- **Advanced ML Models**: Deep learning and ensemble methods
- **Web Dashboard**: Interactive Streamlit/Dash application
- **Mobile App**: On-the-go market insights

### Analytics Expansion
- **Multi-city Analysis**: Comparative market studies
- **Predictive Forecasting**: Long-term market trend predictions
- **Sentiment Analysis**: Social media and news impact on prices
- **Economic Integration**: Macro-economic factor correlation

## 🤝 Contributing

We welcome contributions to enhance this project! Areas for contribution:
- Additional data sources and features
- Advanced machine learning models
- Enhanced visualizations
- Performance optimizations

## 🏆 Acknowledgments

- **City of Edmonton**: For providing comprehensive open data access
- **Academic Supervision**: Dr. Wali Abdullah for guidance and review
- **Python Community**: For excellent data science libraries and tools

## 👥 Project Team

### 👨‍💻 Developer & Analyst
**Devraj Parmar**
- 🌐 [GitHub](https://github.com/17devraj)
- 💼 [LinkedIn](https://linkedin.com/in/devraj-parmar)

### 👨‍🏫 Academic Supervisor
**Dr. Wali Abdullah**
- 🌐 [GitHub](https://github.com/WaliAbdullah)
- 💼 [LinkedIn](https://www.linkedin.com/in/wali-mohammad-abdullah/)

---

**⭐ If you find this project helpful, please consider giving it a star!**

*Last Updated: July 2025*
