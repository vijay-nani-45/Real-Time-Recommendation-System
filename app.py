import streamlit as st
from model import RecommendationSystem

def initialize_session_state():
    if 'rec_system' not in st.session_state:
        st.session_state.rec_system = RecommendationSystem()

def display_product_details(product_details):
    st.subheader("Selected Product Details")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Name:** {product_details['name']}")
        st.write(f"**Category:** {product_details['category']}")
    with col2:
        st.write(f"**Price:** ${product_details['price']:.2f}")
        st.write(f"**Product ID:** {product_details['product_id']}")
    st.write(f"**Description:** {product_details['description']}")

def display_recommendations(recommendations):
    st.header("Recommended Products")
    for _, rec in recommendations.iterrows():
        with st.expander(f"{rec['name']} - ${rec['price']:.2f}"):
            st.write(f"**Category:** {rec['category']}")
            st.write(f"**Description:** {rec['description']}")

def main():
    st.title("E-commerce Recommendation System")
    
    # Initialize session state
    initialize_session_state()
    
    # Browsing history section
    st.header("Browse Products")
    
    # Category filter
    selected_category = st.selectbox(
        "Filter by Category",
        options=['All'] + st.session_state.rec_system.get_categories()
    )
    
    # Get filtered products
    filtered_df = st.session_state.rec_system.get_products_by_category(selected_category)
    
    # Product selection
    selected_product = st.selectbox(
        "Select a Product",
        options=filtered_df['name'].tolist(),
        key='product_selector'
    )
    
    # Display product details
    product_details = st.session_state.rec_system.get_product_by_name(selected_product)
    display_product_details(product_details)
    
    # Get and display recommendations
    recommendations = st.session_state.rec_system.get_recommendations(product_details['product_id'])
    display_recommendations(recommendations)

if __name__ == "__main__":
    main()