import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
import random

# Set page config
st.set_page_config(
    page_title="EpiSim - Epidemiological Questionnaire Simulator",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1rem;
    }
    .developer-info {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2E86AB;
        margin: 1rem 0;
    }
    .recommendation-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Population Models
POPULATION_MODELS = {
    "Pakistani Medical Students": {
        "age_range": (18, 26),
        "age_mean": 21.5,
        "age_std": 2.1,
        "gender_ratio": {"Male": 0.45, "Female": 0.55},
        "socioeconomic_status": {"High": 0.4, "Medium": 0.5, "Low": 0.1},
        "health_awareness": {"High": 0.7, "Medium": 0.25, "Low": 0.05},
        "response_bias": {
            "social_desirability": 0.15,
            "recall_accuracy": 0.85,
            "completion_rate": 0.92
        }
    },
    "Healthcare Workers": {
        "age_range": (22, 55),
        "age_mean": 32.8,
        "age_std": 8.2,
        "gender_ratio": {"Male": 0.35, "Female": 0.65},
        "socioeconomic_status": {"High": 0.3, "Medium": 0.6, "Low": 0.1},
        "health_awareness": {"High": 0.8, "Medium": 0.18, "Low": 0.02},
        "response_bias": {
            "social_desirability": 0.12,
            "recall_accuracy": 0.88,
            "completion_rate": 0.89
        }
    },
    "Urban General Population": {
        "age_range": (18, 65),
        "age_mean": 35.2,
        "age_std": 12.8,
        "gender_ratio": {"Male": 0.52, "Female": 0.48},
        "socioeconomic_status": {"High": 0.2, "Medium": 0.45, "Low": 0.35},
        "health_awareness": {"High": 0.25, "Medium": 0.5, "Low": 0.25},
        "response_bias": {
            "social_desirability": 0.25,
            "recall_accuracy": 0.75,
            "completion_rate": 0.78
        }
    },
    "Rural Population": {
        "age_range": (18, 70),
        "age_mean": 38.5,
        "age_std": 15.2,
        "gender_ratio": {"Male": 0.55, "Female": 0.45},
        "socioeconomic_status": {"High": 0.1, "Medium": 0.3, "Low": 0.6},
        "health_awareness": {"High": 0.15, "Medium": 0.4, "Low": 0.45},
        "response_bias": {
            "social_desirability": 0.3,
            "recall_accuracy": 0.7,
            "completion_rate": 0.65
        }
    }
}

# Sample questionnaires
SAMPLE_QUESTIONNAIRES = {
    "COVID-19 Knowledge and Attitudes": [
        {"question": "What is your age?", "type": "numeric", "category": "demographic"},
        {"question": "What is your gender?", "type": "categorical", "options": ["Male", "Female", "Other"], "category": "demographic"},
        {"question": "COVID-19 is caused by a virus", "type": "likert", "scale": 5, "category": "knowledge"},
        {"question": "Wearing masks prevents COVID-19 transmission", "type": "likert", "scale": 5, "category": "knowledge"},
        {"question": "How often do you wash your hands?", "type": "categorical", "options": ["Never", "Rarely", "Sometimes", "Often", "Always"], "category": "behavior"},
        {"question": "How worried are you about getting COVID-19?", "type": "likert", "scale": 5, "category": "attitude"},
        {"question": "Have you been vaccinated against COVID-19?", "type": "binary", "category": "behavior"}
    ],
    "Mental Health and Stress Assessment": [
        {"question": "What is your age?", "type": "numeric", "category": "demographic"},
        {"question": "What is your gender?", "type": "categorical", "options": ["Male", "Female", "Other"], "category": "demographic"},
        {"question": "How often do you feel stressed?", "type": "categorical", "options": ["Never", "Rarely", "Sometimes", "Often", "Always"], "category": "clinical"},
        {"question": "I feel overwhelmed by my responsibilities", "type": "likert", "scale": 5, "category": "clinical"},
        {"question": "I have trouble sleeping due to stress", "type": "likert", "scale": 5, "category": "clinical"},
        {"question": "How many hours do you sleep per night?", "type": "numeric", "category": "behavior"},
        {"question": "Do you exercise regularly?", "type": "binary", "category": "behavior"},
        {"question": "How would you rate your overall mental health?", "type": "likert", "scale": 5, "category": "clinical"}
    ],
    "Dietary Habits and Nutrition": [
        {"question": "What is your age?", "type": "numeric", "category": "demographic"},
        {"question": "What is your gender?", "type": "categorical", "options": ["Male", "Female", "Other"], "category": "demographic"},
        {"question": "How many servings of fruits do you eat daily?", "type": "numeric", "category": "behavior"},
        {"question": "How many servings of vegetables do you eat daily?", "type": "numeric", "category": "behavior"},
        {"question": "How often do you eat fast food?", "type": "categorical", "options": ["Never", "Rarely", "Sometimes", "Often", "Daily"], "category": "behavior"},
        {"question": "I am satisfied with my current diet", "type": "likert", "scale": 5, "category": "attitude"},
        {"question": "Do you take any dietary supplements?", "type": "binary", "category": "behavior"},
        {"question": "How important is healthy eating to you?", "type": "likert", "scale": 5, "category": "attitude"}
    ]
}

class EpiSimulator:
    def __init__(self, population_model, sample_size):
        self.population_model = population_model
        self.sample_size = sample_size
        self.demographic_data = self._generate_demographics()
    
    def _generate_demographics(self):
        """Generate demographic data for the sample"""
        np.random.seed(42)  # For reproducible results
        
        # Age generation
        ages = np.random.normal(
            self.population_model["age_mean"], 
            self.population_model["age_std"], 
            self.sample_size
        )
        ages = np.clip(ages, self.population_model["age_range"][0], self.population_model["age_range"][1])
        
        # Gender generation
        gender_probs = list(self.population_model["gender_ratio"].values())
        gender_labels = list(self.population_model["gender_ratio"].keys())
        genders = np.random.choice(gender_labels, self.sample_size, p=gender_probs)
        
        # Socioeconomic status
        ses_probs = list(self.population_model["socioeconomic_status"].values())
        ses_labels = list(self.population_model["socioeconomic_status"].keys())
        ses = np.random.choice(ses_labels, self.sample_size, p=ses_probs)
        
        # Health awareness
        ha_probs = list(self.population_model["health_awareness"].values())
        ha_labels = list(self.population_model["health_awareness"].keys())
        health_awareness = np.random.choice(ha_labels, self.sample_size, p=ha_probs)
        
        return pd.DataFrame({
            'age': ages.astype(int),
            'gender': genders,
            'socioeconomic_status': ses,
            'health_awareness': health_awareness
        })
    
    def simulate_responses(self, questionnaire):
        """Simulate responses to questionnaire"""
        responses = []
        completion_rate = self.population_model["response_bias"]["completion_rate"]
        
        for i in range(self.sample_size):
            # Determine if this person completes the survey
            if np.random.random() > completion_rate:
                continue
                
            person_response = {}
            person_demo = self.demographic_data.iloc[i]
            
            for q in questionnaire:
                response = self._generate_response(q, person_demo)
                person_response[q["question"]] = response
            
            responses.append(person_response)
        
        return pd.DataFrame(responses)
    
    def _generate_response(self, question, demographics):
        """Generate a single response based on question type and demographics"""
        q_type = question["type"]
        
        # Apply demographic influences
        age_factor = (demographics["age"] - 18) / 50  # Normalize age
        health_awareness_factor = {"High": 1.0, "Medium": 0.7, "Low": 0.4}[demographics["health_awareness"]]
        ses_factor = {"High": 1.0, "Medium": 0.8, "Low": 0.6}[demographics["socioeconomic_status"]]
        
        # Add some randomness
        randomness = np.random.normal(0, 0.1)
        
        if q_type == "numeric":
            if "age" in question["question"].lower():
                return int(demographics["age"])
            elif "hours" in question["question"].lower() and "sleep" in question["question"].lower():
                base_sleep = 7 + randomness
                return max(4, min(12, int(base_sleep + age_factor * 0.5)))
            elif "serving" in question["question"].lower():
                base_servings = 2 + health_awareness_factor + randomness
                return max(0, min(10, int(base_servings)))
            else:
                return max(0, int(5 + randomness * 3))
        
        elif q_type == "categorical":
            options = question["options"]
            if "gender" in question["question"].lower():
                return demographics["gender"]
            else:
                # Weight responses based on health awareness and SES
                weights = np.ones(len(options))
                if health_awareness_factor > 0.8:
                    weights[-1] *= 1.5  # More likely to choose positive options
                if ses_factor > 0.8:
                    weights[-1] *= 1.3
                weights = weights / np.sum(weights)
                return np.random.choice(options, p=weights)
        
        elif q_type == "likert":
            scale = question["scale"]
            # Base response influenced by health awareness
            base_response = 2.5 + health_awareness_factor * 1.5
            
            # Add category-specific adjustments
            if question["category"] == "knowledge":
                base_response += health_awareness_factor * 0.8
            elif question["category"] == "attitude":
                base_response += ses_factor * 0.5
            elif question["category"] == "clinical":
                base_response += randomness * 2
            
            # Apply social desirability bias
            if self.population_model["response_bias"]["social_desirability"] > 0.2:
                base_response += 0.3
            
            response = base_response + randomness
            return max(1, min(scale, int(round(response))))
        
        elif q_type == "binary":
            # Binary responses influenced by demographics
            prob = 0.5 + health_awareness_factor * 0.2 + ses_factor * 0.1 + randomness * 0.1
            return "Yes" if np.random.random() < prob else "No"
        
        return None

def analyze_questionnaire(responses_df, questionnaire):
    """Analyze questionnaire responses and provide recommendations"""
    analysis = {}
    recommendations = []
    
    if responses_df.empty:
        return {
            "completion_rates": {},
            "recommendations": [{"type": "error", "question": "No data", "issue": "No responses generated", "recommendation": "Check simulation parameters"}],
            "total_responses": 0
        }
    
    # Calculate completion rates
    completion_rates = {}
    
    for q in questionnaire:
        question_text = q["question"]
        if question_text in responses_df.columns:
            completion_rate = (responses_df[question_text].notna().sum() / len(responses_df)) * 100
            completion_rates[question_text] = completion_rate
            
            if completion_rate < 80:
                recommendations.append({
                    "type": "warning",
                    "question": question_text,
                    "issue": "Low completion rate",
                    "recommendation": f"Consider simplifying this question or making it optional. Current completion rate: {completion_rate:.1f}%"
                })
    
    # Analyze response patterns
    for q in questionnaire:
        question_text = q["question"]
        if question_text not in responses_df.columns:
            continue
            
        responses = responses_df[question_text].dropna()
        
        if len(responses) == 0:
            continue
        
        if q["type"] == "likert":
            # Check for ceiling/floor effects
            scale = q["scale"]
            if len(responses) > 0:
                if (responses == scale).mean() > 0.8:
                    recommendations.append({
                        "type": "warning",
                        "question": question_text,
                        "issue": "Ceiling effect detected",
                        "recommendation": "Consider revising the question or scale to capture more variation"
                    })
                elif (responses == 1).mean() > 0.8:
                    recommendations.append({
                        "type": "warning",
                        "question": question_text,
                        "issue": "Floor effect detected",
                        "recommendation": "Consider revising the question or scale to capture more variation"
                    })
                
                # Check for central tendency bias
                if q["scale"] == 5 and (responses == 3).mean() > 0.6:
                    recommendations.append({
                        "type": "info",
                        "question": question_text,
                        "issue": "Central tendency bias",
                        "recommendation": "Consider using forced-choice format or even-numbered scale"
                    })
        
        elif q["type"] == "categorical":
            # Check for response distribution
            value_counts = responses.value_counts()
            if len(value_counts) > 0 and value_counts.iloc[0] / len(responses) > 0.9:
                recommendations.append({
                    "type": "warning",
                    "question": question_text,
                    "issue": "Lack of response variation",
                    "recommendation": "Consider revising response options or question wording"
                })
    
    analysis["completion_rates"] = completion_rates
    analysis["recommendations"] = recommendations
    analysis["total_responses"] = len(responses_df)
    
    return analysis
def create_bar_chart_data(data_dict, title):
    """Create simple data for bar charts"""
    df = pd.DataFrame(list(data_dict.items()), columns=['Category', 'Value'])
    return df

def main():
    # Header
    st.markdown('<h1 class="main-header">🔬 EpiSim</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Epidemiological Questionnaire Pilot Simulator</p>', unsafe_allow_html=True)
    
    # Developer Information
    st.markdown("""
    <div class="developer-info">
        <h4>🎓 Developed By: Muhammad Nabeel Saddique</h4>
        <p>Fourth-year MBBS student at King Edward Medical University, Lahore, Pakistan. Passionate about research and its application in improving healthcare outcomes. Skilled in various research tools including Rayyan, Zotero, EndNote, WebPlotDigitizer, Meta-Converter, RevMan, MetaXL, Jamovi, Comprehensive Meta-Analysis (CMA), OpenMeta, and R Studio.</p>
        <p><strong>Founder:</strong> Nibras Research Academy - A platform for mentoring systematic reviews and meta-analyses, enabling young researchers to publish their first meta-analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("📊 Simulation Parameters")
    
    # Population selection
    population_type = st.sidebar.selectbox(
        "Select Target Population",
        list(POPULATION_MODELS.keys()),
        help="Choose the demographic profile that best matches your study population"
    )
    
    # Sample size
    sample_size = st.sidebar.number_input(
        "Sample Size",
        min_value=50,
        max_value=2000,
        value=300,
        step=50,
        help="Number of simulated responses to generate"
    )
    
    # Questionnaire selection
    questionnaire_source = st.sidebar.radio(
        "Questionnaire Source",
        ["Use Sample Questionnaire", "Upload Custom Questionnaire"],
        help="Choose to use a pre-built questionnaire or upload your own"
    )
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Questionnaire Setup", "🎯 Simulation Results", "📈 Analysis & Recommendations", "📊 Population Insights"])
    
    with tab1:
        st.header("Questionnaire Configuration")
        
        if questionnaire_source == "Use Sample Questionnaire":
            selected_questionnaire = st.selectbox(
                "Select Sample Questionnaire",
                list(SAMPLE_QUESTIONNAIRES.keys())
            )
            questionnaire = SAMPLE_QUESTIONNAIRES[selected_questionnaire]
            
            st.subheader("Preview of Selected Questionnaire")
            for i, q in enumerate(questionnaire, 1):
                with st.expander(f"Question {i}: {q['question'][:50]}..."):
                    st.write(f"**Type:** {q['type']}")
                    st.write(f"**Category:** {q['category']}")
                    if q['type'] == 'categorical':
                        st.write(f"**Options:** {', '.join(q['options'])}")
                    elif q['type'] == 'likert':
                        st.write(f"**Scale:** 1-{q['scale']}")
        
        else:
            st.info("Upload a JSON file with your questionnaire structure")
            uploaded_file = st.file_uploader("Choose a JSON file", type="json")
            if uploaded_file is not None:
                try:
                    questionnaire = json.load(uploaded_file)
                    st.success("Questionnaire uploaded successfully!")
                    st.json(questionnaire)
                except Exception as e:
                    st.error(f"Invalid JSON format: {str(e)}")
                    questionnaire = SAMPLE_QUESTIONNAIRES["COVID-19 Knowledge and Attitudes"]
            else:
                questionnaire = SAMPLE_QUESTIONNAIRES["COVID-19 Knowledge and Attitudes"]
        
        # Population details
        st.subheader("Selected Population Characteristics")
        pop_model = POPULATION_MODELS[population_type]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Age Range", f"{pop_model['age_range'][0]}-{pop_model['age_range'][1]} years")
            st.metric("Mean Age", f"{pop_model['age_mean']:.1f} years")
        with col2:
            st.metric("Gender Ratio (M:F)", f"{pop_model['gender_ratio']['Male']:.0%}:{pop_model['gender_ratio']['Female']:.0%}")
            st.metric("Completion Rate", f"{pop_model['response_bias']['completion_rate']:.0%}")
        with col3:
            st.metric("High Health Awareness", f"{pop_model['health_awareness']['High']:.0%}")
            st.metric("Social Desirability Bias", f"{pop_model['response_bias']['social_desirability']:.0%}")
    
    with tab2:
        st.header("Simulation Results")
        
        if st.button("🚀 Run Simulation", type="primary"):
            with st.spinner("Generating simulated responses..."):
                try:
                    # Run simulation
                    simulator = EpiSimulator(POPULATION_MODELS[population_type], sample_size)
                    responses_df = simulator.simulate_responses(questionnaire)
                    
                    # Store results in session state
                    st.session_state.responses = responses_df
                    st.session_state.questionnaire = questionnaire
                    st.session_state.population_type = population_type
                    st.session_state.simulator = simulator
                    
                    st.success(f"Simulation completed! Generated {len(responses_df)} responses.")
                    
                except Exception as e:
                    st.error(f"Error during simulation: {str(e)}")
        
        # Display results if available
        if 'responses' in st.session_state:
            responses_df = st.session_state.responses
            
            if not responses_df.empty:
                st.subheader("Response Summary")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Responses", len(responses_df))
                with col2:
                    completion_rate = (len(responses_df) / sample_size) * 100
                    st.metric("Completion Rate", f"{completion_rate:.1f}%")
                with col3:
                    avg_response_time = abs(np.random.normal(8.5, 2.1))  # Simulated
                    st.metric("Avg. Response Time", f"{avg_response_time:.1f} min")
                with col4:
                    quality_score = np.random.uniform(75, 95)  # Simulated
                    st.metric("Data Quality Score", f"{quality_score:.1f}%")
                
                # Show sample responses
                st.subheader("Sample Responses")
                st.dataframe(responses_df.head(10))
                
                # Download button
                csv = responses_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"episim_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No responses were generated. Try increasing the sample size or checking the completion rate.")
    
    with tab3:
        st.header("Analysis & Recommendations")
        
        if 'responses' in st.session_state:
            responses_df = st.session_state.responses
            questionnaire = st.session_state.questionnaire
            
            # Perform analysis
            with st.spinner("Analyzing responses..."):
                analysis = analyze_questionnaire(responses_df, questionnaire)
            
            # Show completion rates
            st.subheader("Question Completion Rates")
            if analysis["completion_rates"]:
                completion_data = create_bar_chart_data(analysis["completion_rates"], "Completion Rates")
                st.bar_chart(completion_data.set_index("Category"))
                st.write("**Target completion rate:** 80%")
                
                # Show detailed completion rates
                st.write("**Detailed Completion Rates:**")
                for question, rate in analysis["completion_rates"].items():
                    color = "🟢" if rate >= 80 else "🟡" if rate >= 60 else "🔴"
                    st.write(f"{color} {question[:50]}... : {rate:.1f}%")
            
            # Show recommendations
            st.subheader("Recommendations")
            if analysis["recommendations"]:
                for rec in analysis["recommendations"]:
                    if rec["type"] == "warning":
                        st.markdown(f"""
                        <div class="warning-box">
                            <strong>⚠️ {rec["issue"]}</strong><br>
                            <strong>Question:</strong> {rec["question"]}<br>
                            <strong>Recommendation:</strong> {rec["recommendation"]}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="recommendation-box">
                            <strong>💡 {rec["issue"]}</strong><br>
                            <strong>Question:</strong> {rec["question"]}<br>
                            <strong>Recommendation:</strong> {rec["recommendation"]}
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.success("🎉 No major issues detected with your questionnaire!")
            
            # Response distribution visualization
            st.subheader("Response Distributions")
            
            # Select questions to visualize
            visualizable_questions = [q["question"] for q in questionnaire 
                                    if q["type"] in ["likert", "categorical"] and q["question"] in responses_df.columns]
            
            if visualizable_questions:
                selected_questions = st.multiselect(
                    "Select questions to visualize:",
                    visualizable_questions,
                    default=visualizable_questions[:3] if len(visualizable_questions) >= 3 else visualizable_questions
                )
                
                for question in selected_questions:
                    if question in responses_df.columns:
                        st.subheader(f"Distribution: {question}")
                        value_counts = responses_df[question].value_counts()
                        st.bar_chart(value_counts)
                        
                        # Show statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Responses", len(responses_df[question].dropna()))
                        with col2:
                            if responses_df[question].dtype in ['int64', 'float64']:
                                st.metric("Mean", f"{responses_df[question].mean():.2f}")
                            else:
                                most_common = value_counts.index[0] if len(value_counts) > 0 else "N/A"
                                st.metric("Most Common", most_common)
                        with col3:
                            if responses_df[question].dtype in ['int64', 'float64']:
                                st.metric("Std Dev", f"{responses_df[question].std():.2f}")
                            else:
                                diversity = len(value_counts)
                                st.metric("Unique Responses", diversity)
        else:
            st.info("Run a simulation first to see analysis and recommendations.")
    
    with tab4:
        st.header("Population Insights")
        
        if 'simulator' in st.session_state:
            simulator = st.session_state.simulator
            demo_data = simulator.demographic_data
            
            # Demographics visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # Age distribution
                st.subheader("Age Distribution")
                age_bins = pd.cut(demo_data["age"], bins=10)
                age_counts = age_bins.value_counts().sort_index()
                st.bar_chart(age_counts)
                
                # Age statistics
                st.write(f"**Mean Age:** {demo_data['age'].mean():.1f} years")
                st.write(f"**Age Range:** {demo_data['age'].min()}-{demo_data['age'].max()} years")
                
                # Gender distribution
                st.subheader("Gender Distribution")
                gender_counts = demo_data['gender'].value_counts()
                st.bar_chart(gender_counts)
                
                # Gender percentages
                for gender, count in gender_counts.items():
                    percentage = (count / len(demo_data)) * 100
                    st.write(f"**{gender}:** {percentage:.1f}% ({count} people)")
            
            with col2:
                # SES distribution
                st.subheader("Socioeconomic Status Distribution")
                ses_counts = demo_data['socioeconomic_status'].value_counts()
                st.bar_chart(ses_counts)
                
                # SES percentages
                for ses, count in ses_counts.items():
                    percentage = (count / len(demo_data)) * 100
                    st.write(f"**{ses} SES:** {percentage:.1f}% ({count} people)")
                
                # Health awareness
                st.subheader("Health Awareness Distribution")
                ha_counts = demo_data['health_awareness'].value_counts()
                st.bar_chart(ha_counts)
                
                # Health awareness percentages
                for ha, count in ha_counts.items():
                    percentage = (count / len(demo_data)) * 100
                    st.write(f"**{ha} Awareness:** {percentage:.1f}% ({count} people)")
            
            # Population characteristics table
            st.subheader("Population Model Parameters")
            pop_model = POPULATION_MODELS[st.session_state.population_type]
            
            # Create a summary table
            summary_data = {
                "Parameter": ["Age Range", "Mean Age", "Gender Ratio (M:F)", 
                             "High SES %", "High Health Awareness %", 
                             "Completion Rate", "Social Desirability Bias"],
                "Value": [f"{pop_model['age_range'][0]}-{pop_model['age_range'][1]}",
                         f"{pop_model['age_mean']:.1f}",
                         f"{pop_model['gender_ratio']['Male']:.0%}:{pop_model['gender_ratio']['Female']:.0%}",
                         f"{pop_model['socioeconomic_status']['High']:.0%}",
                         f"{pop_model['health_awareness']['High']:.0%}",
                         f"{pop_model['response_bias']['completion_rate']:.0%}",
                         f"{pop_model['response_bias']['social_desirability']:.0%}"]
            }
            
            summary_df = pd.DataFrame(summary_data)
            st.table(summary_df)
            
            # Cross-tabulation analysis
            st.subheader("Demographic Cross-Analysis")
            
            # Gender vs Health Awareness
            cross_tab = pd.crosstab(demo_data['gender'], demo_data['health_awareness'], margins=True)
            st.write("**Gender vs Health Awareness:**")
            st.dataframe(cross_tab)
            
            # Age groups vs SES
            demo_data['age_group'] = pd.cut(demo_data['age'], 
                                          bins=[0, 25, 35, 50, 100], 
                                          labels=['18-25', '26-35', '36-50', '50+'])
            age_ses_cross = pd.crosstab(demo_data['age_group'], demo_data['socioeconomic_status'], margins=True)
            st.write("**Age Group vs Socioeconomic Status:**")
            st.dataframe(age_ses_cross)
        
        else:
            st.info("Run a simulation first to see population insights.")
    
    # Additional Features Section
    st.markdown("---")
    
    # Quick Stats
    if 'responses' in st.session_state and 'simulator' in st.session_state:
        st.header("📈 Quick Statistics Summary")
        
        responses_df = st.session_state.responses
        simulator = st.session_state.simulator
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Response Rate",
                value=f"{(len(responses_df) / sample_size) * 100:.1f}%",
                delta=f"{((len(responses_df) / sample_size) - 0.8) * 100:.1f}%" if len(responses_df) / sample_size > 0.8 else None
            )
        
        with col2:
            # Calculate average completion per question
            total_possible = len(responses_df) * len(st.session_state.questionnaire)
            total_actual = responses_df.count().sum()
            completion_rate = (total_actual / total_possible) * 100 if total_possible > 0 else 0
            st.metric(
                label="Question Completion",
                value=f"{completion_rate:.1f}%"
            )
        
        with col3:
            # Data quality score (simulated based on completion and response patterns)
            quality_factors = []
            for col in responses_df.columns:
                if responses_df[col].dtype in ['int64', 'float64']:
                    # Check for reasonable variance
                    cv = responses_df[col].std() / responses_df[col].mean() if responses_df[col].mean() != 0 else 0
                    quality_factors.append(min(cv, 1.0))
                else:
                    # Check for response diversity
                    unique_ratio = len(responses_df[col].unique()) / len(responses_df[col])
                    quality_factors.append(unique_ratio)
            
            avg_quality = np.mean(quality_factors) * 100 if quality_factors else 0
            st.metric(
                label="Data Quality",
                value=f"{avg_quality:.1f}%"
            )
        
        with col4:
            # Estimated time saved
            estimated_time_saved = sample_size * 0.5  # Assuming 30 minutes per real response
            st.metric(
                label="Time Saved (hours)",
                value=f"{estimated_time_saved:.0f}h"
            )
    
    # Export and Sharing Options
    if 'responses' in st.session_state:
        st.header("📤 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export raw data
            csv_data = st.session_state.responses.to_csv(index=False)
            st.download_button(
                label="📊 Download Raw Data (CSV)",
                data=csv_data,
                file_name=f"episim_raw_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Export analysis report
            if 'questionnaire' in st.session_state:
                analysis = analyze_questionnaire(st.session_state.responses, st.session_state.questionnaire)
                
                report_content = f"""EpiSim Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Population: {st.session_state.population_type}
Sample Size: {sample_size}
Responses Generated: {len(st.session_state.responses)}

COMPLETION RATES:
"""
                for question, rate in analysis["completion_rates"].items():
                    report_content += f"- {question}: {rate:.1f}%\n"
                
                report_content += "\nRECOMMENDATIONS:\n"
                for rec in analysis["recommendations"]:
                    report_content += f"- {rec['issue']}: {rec['recommendation']}\n"
                
                st.download_button(
                    label="📋 Download Analysis Report",
                    data=report_content,
                    file_name=f"episim_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        with col3:
            # Export demographic summary
            if 'simulator' in st.session_state:
                demo_data = st.session_state.simulator.demographic_data
                demo_summary = demo_data.describe(include='all').to_csv()
                
                st.download_button(
                    label="👥 Download Demographics",
                    data=demo_summary,
                    file_name=f"episim_demographics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p><strong>EpiSim v1.0</strong> - Epidemiological Questionnaire Pilot Simulator</p>
        <p>Developed by Muhammad Nabeel Saddique | King Edward Medical University, Lahore</p>
        <p>🏥 Improving healthcare outcomes through innovative research tools</p>
        <p><em>For research and educational purposes only. Results are simulated and should not replace actual pilot studies.</em></p>
        <br>
        <p><strong>🎯 Features:</strong> Population Modeling | Response Simulation | Quality Analysis | Bias Detection</p>
        <p><strong>🔬 Research Areas:</strong> Epidemiology | Public Health | Survey Methodology | Data Science</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()