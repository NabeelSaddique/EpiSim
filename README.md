# 🔬 EpiSim - Epidemiological Questionnaire Pilot Simulator

A comprehensive Python-based web application for simulating epidemiological study responses, designed to streamline the pilot testing phase of public health research questionnaires.

## 🎓 Developer

**Muhammad Nabeel Saddique**  
Fourth-year MBBS student at King Edward Medical University, Lahore, Pakistan

- 🔬 Passionate about research and its application in improving healthcare outcomes
- 🛠️ Skilled in various research tools: Rayyan, Zotero, EndNote, WebPlotDigitizer, Meta-Converter, RevMan, MetaXL, Jamovi, Comprehensive Meta-Analysis (CMA), OpenMeta, and R Studio
- 🏫 Founder of **Nibras Research Academy** - A platform for mentoring systematic reviews and meta-analyses, enabling young researchers to publish their first meta-analysis

## 📖 Overview

EpiSim addresses a critical need in epidemiological research by providing realistic simulation of questionnaire responses before conducting actual pilot studies. This saves time, resources, and helps researchers optimize their survey instruments for better data quality.

## ✨ Features

### 🎯 Population Models
- **Pakistani Medical Students** (Age 18-26, High health awareness)
- **Healthcare Workers** (Age 22-55, Very high health awareness)
- **Urban General Population** (Age 18-65, Mixed demographics)
- **Rural Population** (Age 18-70, Lower health awareness)

### 📋 Sample Questionnaires
- **COVID-19 Knowledge and Attitudes**
- **Mental Health and Stress Assessment**
- **Dietary Habits and Nutrition**

### 🧮 Simulation Engine
- Realistic demographic generation based on Pakistani population data
- Response bias modeling (social desirability, recall accuracy, completion rates)
- Population-specific response patterns
- Configurable sample sizes (50-2000 participants)

### 📊 Analysis Dashboard
- Real-time response completion rates
- Interactive distribution visualizations
- Statistical reliability assessments
- Data quality metrics

### 💡 Smart Recommendations
- Detects ceiling and floor effects
- Identifies questions with low completion rates
- Flags central tendency bias in Likert scales
- Suggests specific improvements for questionnaire optimization

## 🚀 Quick Start

### Prerequisites
- Python 3.7 or higher
- Streamlit
- Required packages (see requirements.txt)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/episim.git
cd episim
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

4. **Open your browser** to `http://localhost:8501`

### Requirements.txt
```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
plotly>=5.15.0
scipy>=1.10.0
scikit-learn>=1.3.0
```

## 🔧 Usage Guide

### Step 1: Configure Simulation Parameters
1. Select your target population from the sidebar
2. Set the desired sample size (50-2000)
3. Choose between sample questionnaires or upload custom JSON

### Step 2: Review Questionnaire
1. Preview your selected questionnaire
2. Review question types and categories
3. Check population characteristics

### Step 3: Run Simulation
1. Click "Run Simulation" to generate responses
2. View completion rates and summary statistics
3. Download results as CSV for further analysis

### Step 4: Analyze Results
1. Review question completion rates
2. Examine response distributions
3. Read AI-generated recommendations
4. Identify potential questionnaire improvements

## 📁 Project Structure

```
episim/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This documentation
├── data/
│   ├── population_models.json    # Population parameters
│   └── sample_questionnaires.json # Pre-built questionnaires
└── docs/
    ├── user_guide.md     # Detailed user documentation
    └── api_reference.md  # Technical API documentation
```

## 🎨 Custom Questionnaire Format

To upload custom questionnaires, use this JSON structure:

```json
[
  {
    "question": "What is your age?",
    "type": "numeric",
    "category": "demographic"
  },
  {
    "question": "How often do you exercise?",
    "type": "categorical",
    "options": ["Never", "Rarely", "Sometimes", "Often", "Daily"],
    "category": "behavior"
  },
  {
    "question": "I feel confident about my health knowledge",
    "type": "likert",
    "scale": 5,
    "category": "attitude"
  },
  {
    "question": "Do you have health insurance?",
    "type": "binary",
    "category": "demographic"
  }
]
```

### Question Types
- **numeric**: Age, counts, measurements
- **categorical**: Multiple choice with predefined options
- **likert**: Scale responses (1-5 or 1-7)
- **binary**: Yes/No questions

### Categories
- **demographic**: Age, gender, education, etc.
- **knowledge**: Factual knowledge questions
- **attitude**: Opinions and beliefs
- **behavior**: Actions and practices
- **clinical**: Health status and symptoms

## 🌐 Online Deployment

### Streamlit Cloud Deployment

1. **Push to GitHub**
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

2. **Deploy on Streamlit Cloud**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select your repository
   - Deploy automatically

3. **Alternative Deployment Options**
   - **Heroku**: Use `Procfile` for Heroku deployment
   - **AWS EC2**: Deploy on cloud instances
   - **Docker**: Containerized deployment

### Environment Variables
```bash
# For production deployment
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

## 📈 Technical Architecture

### Population Modeling
- **Demographic Generation**: Normal distributions for age, weighted sampling for categorical variables
- **Response Patterns**: Bayesian-influenced responses based on demographic characteristics
- **Bias Simulation**: Social desirability bias, recall accuracy, and completion rate modeling

### Statistical Methods
- **Monte Carlo Simulation**: For generating individual responses
- **Correlation Analysis**: For internal consistency assessment
- **Distribution Analysis**: For detecting response patterns and biases

### Performance Optimization
- **Caching**: Streamlit caching for population models and computations
- **Vectorized Operations**: NumPy and Pandas for efficient data processing
- **Lazy Loading**: On-demand generation of simulation results

## 🧪 Validation and Testing

### Model Validation
- **Literature Comparison**: Population parameters based on published Pakistani demographic data
- **Expert Review**: Validated by epidemiologists and public health researchers
- **Sensitivity Analysis**: Tested across different parameter ranges

### Quality Assurance
- **Unit Tests**: Individual function testing
- **Integration Tests**: End-to-end workflow validation
- **User Testing**: Feedback from medical students and researchers

## 📊 Output Formats

### CSV Export
- Raw response data
- Demographic breakdowns
- Summary statistics
- Quality metrics

### Analysis Reports
- Completion rate analysis
- Response distribution summaries
- Reliability assessments
- Improvement recommendations

## 🤝 Contributing

We welcome contributions from the research community!

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Areas for Contribution
- Additional population models for other countries/regions
- New questionnaire templates
- Enhanced statistical analysis methods
- UI/UX improvements
- Documentation enhancements

## 🙏 Acknowledgments

- **King Edward Medical University** for institutional support
- **Nibras Research Academy** for research methodology insights
- **Streamlit Community** for the excellent web framework
- **Pakistani Health Research Community** for demographic data and validation

## 📞 Contact

**Muhammad Nabeel Saddique**
- 📧 Email: [nabeelsaddique@kemu.edu.pk]
- 🐙 GitHub: [NabeelSaddique]
- 🏥 Institution: King Edward Medical University, Lahore, Pakistan

## 🚨 Disclaimer

EpiSim is designed for research and educational purposes. While the simulation models are based on empirical data and validated methodologies, the results are simulated and should not replace actual pilot studies in formal research protocols. Always conduct appropriate pilot testing before implementing large-scale epidemiological studies.

## 📚 References

1. World Health Organization. (2023). *Guidelines for conducting pilot studies in epidemiological research*
2. Pakistan Bureau of Statistics. (2023). *Demographic and Health Survey*
3. Streamlit Documentation. (2024). *Building interactive web applications*
4. NumPy/Pandas Development Team. (2024). *Scientific computing in Python*

---

**Version**: 1.0  
**Last Updated**: June 2025  
**Minimum Python Version**: 3.7+  
**Streamlit Version**: 1.28.0+
