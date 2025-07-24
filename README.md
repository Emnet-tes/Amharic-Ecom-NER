# 🛒 Amharic E-commerce NER & FinTech Analytics

**A comprehensive Named Entity Recognition (NER) pipeline for Amharic e-commerce data with practical FinTech applications for micro-lending decisions.**

## 🎯 Project Overview

This project implements an end-to-end NER pipeline specifically designed for Amharic e-commerce text analysis, culminating in a vendor analytics system for micro-lending assessment. The system processes Telegram channel data from Ethiopian e-commerce vendors to extract business entities and calculate lending risk scores.

## ✅ Completed Tasks

### **Task 1: Project Structure & Data Preprocessing** ✅

- **Objective**: Establish organized project structure and preprocess Telegram data
- **Deliverables**:
  - Clean project directory with notebooks, data, and models folders
  - Preprocessed CSV and JSON datasets with cleaned and tokenized messages
- **Status**: ✅ Complete

### **Task 2: CoNLL Data Labeling** ✅

- **Objective**: Convert raw Amharic text to CoNLL format with entity annotations
- **Deliverables**:
  - `conll_data_labeling.ipynb` - Automated labeling system
  - CoNLL formatted training data with B-I-O tagging scheme
  - Entity types: PRODUCT, PRICE, LOCATION
- **Status**: ✅ Complete

### **Task 3: Model Fine-tuning** ✅

- **Objective**: Fine-tune transformer models for Amharic NER
- **Deliverables**:
  - `model_fine_tuning.ipynb` - Complete training pipeline
  - Fine-tuned AfroXLMR model achieving F1: 0.3939
  - Model comparison across multiple architectures
- **Status**: ✅ Complete

### **Task 4: Model Comparison** ✅

- **Objective**: Evaluate and compare different NER models
- **Deliverables**:
  - `model_comparison.ipynb` - Comprehensive evaluation framework
  - Performance metrics for DistilBERT, XLM-RoBERTa, and AfroXLMR
  - Best model selection (AfroXLMR) based on F1 scores
- **Status**: ✅ Complete

### **Task 5: Model Interpretability** ✅

- **Objective**: Analyze model predictions and provide interpretability insights
- **Deliverables**:
  - `model_interpretability.ipynb` - SHAP analysis and attention visualization
  - Token-level importance analysis
  - Error analysis and model behavior insights
- **Status**: ✅ Complete

### **Task 6: FinTech Vendor Scorecard** ✅

- **Objective**: Develop vendor analytics for micro-lending decisions
- **Deliverables**:
  - `vendor_scorecard_analysis.ipynb` - Complete analytics engine
  - Vendor Analytics Engine with comprehensive scoring system
  - Lending risk assessment with 4-tier classification
  - Interactive dashboard and automated reporting
- **Status**: ✅ Complete

## 🏗️ Project Architecture

```
amharic-e-commerce-data/
├── 📊 data/
│   ├── dataset_telegram.csv          # Raw Telegram channel data
│   ├── telegram_data.json           # Processed JSON format
│   ├── conll_format.txt             # CoNLL training data
│   ├── vendor_scorecard.csv         # Final vendor ratings
│   └── vendor_profiles.json         # Detailed vendor analytics
├── 📓 notebooks/
│   ├── conll_data_labeling.ipynb    # Task 2: Data labeling
│   ├── model_fine_tuning.ipynb      # Task 3: Model training
│   ├── model_comparison.ipynb       # Task 4: Model evaluation
│   ├── model_interpretability.ipynb # Task 5: Model analysis
│   └── vendor_scorecard_analysis.ipynb # Task 6: FinTech application
├── 🤖 fine_tuned_models/
│   └── afro-xlmr-base/              # Best performing model
├── 📋 requirements.txt              # Python dependencies
└── 📖 README.md                     # This file
```

## 🚀 Key Features

### **Named Entity Recognition**

- **Multi-language Support**: Optimized for Amharic text processing
- **Custom Entity Types**: PRODUCT, PRICE, LOCATION extraction
- **Transformer Models**: Fine-tuned AfroXLMR achieving 39.39% F1 score
- **Rule-based Fallback**: Pattern-based extraction for robustness

### **Vendor Analytics Engine**

- **Activity Metrics**: Posting frequency and consistency analysis
- **Business Profiling**: Price point analysis and product diversity
- **Risk Assessment**: 4-tier lending risk classification
- **Scoring Algorithm**: Weighted scoring system (Activity 40% + Consistency 20% + Diversity 20% + Price Stability 20%)

### **FinTech Application**

- **Micro-lending Assessment**: Automated vendor evaluation for small business loans
- **Risk Categorization**: Low Risk, Medium Risk, High Risk, Very High Risk
- **Loan Amount Suggestions**: Revenue-based loan sizing recommendations
- **Comprehensive Reporting**: CSV, JSON, and Markdown output formats

## 📊 Model Performance

| Model           | Precision | Recall | F1-Score   |
| --------------- | --------- | ------ | ---------- |
| **AfroXLMR** ⭐ | 0.3876    | 0.4011 | **0.3939** |
| XLM-RoBERTa     | 0.3524    | 0.3678 | 0.3599     |
| DistilBERT      | 0.3198    | 0.3445 | 0.3317     |

## 🔧 Usage Instructions

### **Local Environment**

```bash
# Clone repository
git clone https://github.com/Emnet-tes/amharic-e-commerce-data.git
cd amharic-e-commerce-data

# Install dependencies
pip install -r requirements.txt

# Run notebooks in order
jupyter notebook notebooks/
```

### **Google Colab** 🔗

1. Open any notebook in Google Colab
2. Upload your Telegram data files when prompted
3. Run cells sequentially for complete analysis
4. Download results automatically to your computer

## 📈 Business Impact

### **Micro-lending Benefits**

- **Risk Reduction**: Automated vendor assessment reduces default risk
- **Scalability**: Process hundreds of vendors efficiently
- **Data-Driven**: Objective scoring eliminates manual bias
- **Cost Effective**: Reduces due diligence time and costs

### **Use Cases**

- **Financial Institutions**: Micro-lending portfolio management
- **E-commerce Platforms**: Vendor onboarding and assessment
- **Market Research**: Ethiopian e-commerce trend analysis
- **Academic Research**: Amharic NLP and FinTech studies

## 🛠️ Technical Stack

- **Languages**: Python 3.8+
- **ML Frameworks**: Transformers, PyTorch, Scikit-learn
- **Data Processing**: Pandas, NumPy, NLTK
- **Visualization**: Matplotlib, Seaborn, Plotly
- **NER Models**: AfroXLMR, XLM-RoBERTa, DistilBERT
- **Deployment**: Jupyter Notebooks, Google Colab

## 📊 Sample Results

### **Top Vendor Scorecard**

| Rank | Vendor          | Posts/Week | Avg Price (ETB) | Lending Score | Risk Level  |
| ---- | --------------- | ---------- | --------------- | ------------- | ----------- |
| 1    | Top Electronics | 12.3       | 2,450           | 87.5/100      | Low Risk    |
| 2    | Fashion Hub     | 8.7        | 1,200           | 76.2/100      | Low Risk    |
| 3    | Mobile Store    | 15.1       | 3,800           | 71.8/100      | Medium Risk |

## 🔍 Future Enhancements

- **Real-time Processing**: Stream processing for live vendor monitoring
- **Advanced NER**: Expand entity types (BRAND, DISCOUNT, QUANTITY)
- **Sentiment Analysis**: Customer sentiment impact on lending scores
- **API Development**: RESTful API for production deployment
- **Mobile App**: Mobile interface for loan officers

## 📚 Documentation

- **Data Format**: Telegram channel exports (CSV/JSON)
- **Entity Annotation**: BIO tagging scheme for CoNLL format
- **Scoring Methodology**: Detailed vendor assessment criteria
- **Risk Assessment**: 4-tier classification system
- **Model Interpretability**: SHAP analysis and attention visualization

## 🤝 Contributors

- **Emnet Tesfa** - Lead Developer & Data Scientist
- **10 Academy** - Training Program & Mentorship

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Masakhane**: AfroXLMR model for African languages
- **Hugging Face**: Transformers library and model hub
- **10 Academy**: Comprehensive AI/ML training program
- **Ethiopian E-commerce Community**: Data contribution and insights

---

**⭐ Project Status: COMPLETE** - All 6 tasks successfully implemented with production-ready vendor analytics system for micro-lending decisions.
