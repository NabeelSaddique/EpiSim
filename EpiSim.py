import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import io
from datetime import datetime
import random
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="EpiSim - Epidemiological Questionnaire Simulator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .developer-info {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2E86AB;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
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
        "education_level": "University",
        "socioeconomic_status": {"High": 0.4, "Medium": 0.5, "Low": 0.1},
        "health_awareness": {"High": 0.7, "Medium": 0.25, "Low": 0.05},
        "response_bias": {
            "social_desirability": 0.15,
            "recall_accuracy": 0.85,
            "completion_rate": 0.92
        },
        "health_behaviors": {
            "exercise_regularly": 0.35,
            "healthy_diet": 0.45,
            "smoking": 0.08,
            "stress_levels_high": 0.6
        }
    },
    "Healthcare Workers": {
        "age_range": (22, 55),
        "age_mean": 32.8,
        "age_std": 8.2,
        "gender_ratio": {"Male": 0.35, "Female": 0.65},
        "education_level": "University+",
        "socioeconomic_status": {"High": 0.3, "Medium": 0.6, "Low": 0.1},
        "health_awareness": {"High": 0.8, "Medium": 0.18, "Low": 0.02},
        "response_bias": {
            "social_desirability": 0.12,
            "recall_accuracy": 0.88,
            "completion_rate": 0.89
        },
        "health_behaviors": {
            "exercise_regularly": 0.28,
            "healthy_diet": 0.52,
            "smoking": 0.12,
            "stress_levels_high": 0.7
        }
    },
    "Urban General Population": {
        "age_range": (18, 65),
        "age_mean": 35.2,
        "age_std": 12.8,
        "gender_ratio": {"Male": 0.52, "Female": 0.48},
        "education_level": "Mixed",
        "socioeconomic_status": {"High": 0.2, "Medium": 0.45, "Low": 0.35},
        "health_awareness": {"High": 0.25, "Medium": 0.5, "Low": 0.25},
        "response_bias": {
            "social_desirability": 0.25,
            "recall_accuracy": 0.75,
            "completion_rate": 0.78
        },
        "health_behaviors": {
            "exercise_regularly": 0.22,
            "healthy_diet": 0.35,
            "smoking": 0.18,
            "stress_levels_high": 0.45
        }
    },
    "Rural Population": {
        "age_range": (18, 70),
        "age_mean": 38.5,
        "age_std": 15.2,
        "gender_ratio": {"Male": 0.55, "Female": 0.45},
        "education_level": "Primary/Secondary",
        "socioeconomic_status": {"High": 0.1, "Medium": 0.3, "Low": 0.6},
        "health_awareness": {"High": 0.15, "Medium": 0.4, "Low": 0.45},
        "response_bias": {
            "social_desirability": 0.3,
            "recall_accuracy": 0.7,
            "completion_rate": 0.65
        },
        "health_behaviors": {
            "exercise_regularly": 0.4,
            "healthy_diet": 0.25,
            "smoking": 0.25,
            "stress_levels_high": 0.35
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
            'age': ages,
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
    
    # Calculate completion rates
    total_questions = len(questionnaire)
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
        
        if q["type"] == "likert":
            # Check for ceiling/floor effects
            scale = q["scale"]
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
    
    # Calculate reliability for likert scales
    likert_questions = [q["question"] for q in questionnaire if q["type"] == "likert"]
    if len(likert_questions) > 1:
        likert_data = responses_df[likert_questions].dropna()
        if len(likert_data) > 10:
            # Simple reliability estimate
            correlations = likert_data.corr()
            avg_correlation = correlations.mean().mean()
            if avg_correlation < 0.3:
                recommendations.append({
                    "type": "info",
                    "question": "Overall questionnaire",
                    "issue": "Low internal consistency",
                    "recommendation": "Consider reviewing question relevance and wording for better coherence"
                })
    
    analysis["completion_rates"] = completion_rates
    analysis["recommendations"] = recommendations
    analysis["total_responses"] = len(responses_df)
    
    return analysis

def main():
    # Header
    st.markdown('<h1 class="main-header">🔬 EpiSim</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Epidemiological Questionnaire Pilot Simulator</p>', unsafe_allow_html=True)
    
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
                except:
                    st.error("Invalid JSON format")
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
                # Run simulation
                simulator = EpiSimulator(POPULATION_MODELS[population_type], sample_size)
                responses_df = simulator.simulate_responses(questionnaire)
                
                # Store results in session state
                st.session_state.responses = responses_df
                st.session_state.questionnaire = questionnaire
                st.session_state.population_type = population_type
                st.session_state.simulator = simulator
            
            st.success(f"Simulation completed! Generated {len(responses_df)} responses.")
        
        # Display results if available
        if 'responses' in st.session_state:
            responses_df = st.session_state.responses
            
            st.subheader("Response Summary")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Responses", len(responses_df))
            with col2:
                completion_rate = (len(responses_df) / sample_size) * 100
                st.metric("Completion Rate", f"{completion_rate:.1f}%")
            with col3:
                avg_response_time = np.random.normal(8.5, 2.1)  # Simulated
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
            completion_data = pd.DataFrame(list(analysis["completion_rates"].items()), 
                                         columns=["Question", "Completion Rate"])
            
            fig = px.bar(completion_data, x="Completion Rate", y="Question", 
                        orientation='h', title="Completion Rate by Question")
            fig.add_vline(x=80, line_dash="dash", line_color="red", 
                         annotation_text="Target: 80%")
            st.plotly_chart(fig, use_container_width=True)
            
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
                                    if q["type"] in ["likert", "categorical"]]
            
            if visualizable_questions:
                selected_questions = st.multiselect(
                    "Select questions to visualize:",
                    visualizable_questions,
                    default=visualizable_questions[:3]
                )
                
                for question in selected_questions:
                    if question in responses_df.columns:
                        fig = px.histogram(responses_df, x=question, 
                                         title=f"Distribution: {question}")
                        st.plotly_chart(fig, use_container_width=True)
        
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
                fig = px.histogram(demo_data, x="age", nbins=20, 
                                 title="Age Distribution")
                st.plotly_chart(fig, use_container_width=True)
                
                # Gender distribution
                gender_counts = demo_data['gender'].value_counts()
                fig = px.pie(values=gender_counts.values, names=gender_counts.index,
                           title="Gender Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # SES distribution
                ses_counts = demo_data['socioeconomic_status'].value_counts()
                fig = px.bar(x=ses_counts.index, y=ses_counts.values,
                           title="Socioeconomic Status Distribution")
                st.plotly_chart(fig, use_container_width=True)
                
                # Health awareness
                ha_counts = demo_data['health_awareness'].value_counts()
                fig = px.bar(x=ha_counts.index, y=ha_counts.values,
                           title="Health Awareness Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
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
            
            st.table(pd.DataFrame(summary_data))
        
        else:
            st.info("Run a simulation first to see population insights.")
     # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p><strong>EpiSim v1.0</strong> - Epidemiological Questionnaire Pilot Simulator</p>
        <p>Developed by Muhammad Nabeel Saddique | King Edward Medical University, Lahore</p>
        <p>🏥 Improving healthcare outcomes through innovative research tools</p>
        <p><em>For research and educational purposes only. Results are simulated and should not replace actual pilot studies.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()