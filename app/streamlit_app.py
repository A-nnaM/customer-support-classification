import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from predict import SupportTicketPredictor

# Page configuration
st.set_page_config(
    page_title="Support Ticket Classifier",
    page_icon="🎫",
    layout="centered"
)

# Title and description
st.title("🎫 Customer Support Ticket Classifier")
st.markdown("""
This app classifies customer support tickets into categories using a fine-tuned BERT model.
Enter a support ticket below to get predictions.
""")

# Initialize predictor (cached to avoid reloading)
@st.cache_resource
def load_predictor():
    """Load the model (cached for performance)"""
    with st.spinner("Loading model... (this may take a minute)"):
        predictor = SupportTicketPredictor()
    return predictor

try:
    predictor = load_predictor()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Input section
st.subheader("📝 Enter Support Ticket")

# Example tickets
example_tickets = {
    "Select an example...": "",
    "Billing Issue": "I was charged twice for my subscription this month. Can you refund one charge?",
    "Technical Problem": "The app keeps crashing when I try to login. Error code 500.",
    "Account Access": "I can't access my account after changing my password.",
    "Product Inquiry": "Do you have this product available in blue color?",
    "Shipping Concern": "My order hasn't arrived yet. It's been 2 weeks since I ordered.",
}

# Example selector
selected_example = st.selectbox("Or try an example:", list(example_tickets.keys()))

# Text input
if selected_example != "Select an example...":
    default_text = example_tickets[selected_example]
else:
    default_text = ""

user_input = st.text_area(
    "Support Ticket Text:",
    value=default_text,
    height=150,
    placeholder="Enter customer support ticket text here..."
)

# Threshold slider
threshold = st.slider(
    "Classification Threshold:",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Minimum probability to assign a category"
)

# Predict button
if st.button("🔍 Classify Ticket", type="primary"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text first!")
    else:
        with st.spinner("Analyzing ticket..."):
            # Get prediction
            result = predictor.predict(user_input, threshold=threshold)
        
        # Display results
        st.subheader("📊 Classification Results")
        
        # Predicted categories
        if result['predicted_categories']:
            st.markdown("**🏷️ Predicted Categories:**")
            for category in result['predicted_categories']:
                st.success(f"✓ {category}")
        else:
            st.info("No categories predicted above threshold.")
        
        # All probabilities
        st.markdown("**📈 All Category Probabilities:**")
        
        # Sort by probability
        sorted_probs = sorted(
            result['probabilities'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Display as bars
        for category, prob in sorted_probs:
            # Create visual bar
            bar_length = int(prob * 100)
            bar = "█" * (bar_length // 5)
            
            # Color based on threshold
            if prob >= threshold:
                st.markdown(f"**{category}**: {prob:.3f} 🟢 {bar}")
            else:
                st.markdown(f"{category}: {prob:.3f} ⚪ {bar}")
        
        # Threshold note
        st.caption(f"🟢 = Above threshold ({threshold}) | ⚪ = Below threshold")

# Footer
st.markdown("---")
st.markdown("""
**Model Information:**
- Base Model: BERT (bert-base-uncased)
- Categories: Billing Issue, Technical Problem, Account Access, Product Inquiry, Refund Request, Shipping Concern, Service Complaint
- Task: Multi-label Text Classification
""")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This application uses a BERT-based neural network to classify customer support tickets.
    
    **How to use:**
    1. Enter a support ticket text
    2. Adjust threshold if needed
    3. Click "Classify Ticket"
    4. View predicted categories
    
    **Tips:**
    - Threshold controls sensitivity
    - Lower threshold = more categories
    - Higher threshold = fewer, more confident categories
    """)
    
    st.header("📊 Model Stats")
    st.metric("Categories", len(predictor.label_classes))
    st.metric("Model Size", "~420 MB")
    st.metric("Base Model", "BERT")