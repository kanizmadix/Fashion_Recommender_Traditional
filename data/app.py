import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional
from dataclasses import dataclass
import os

# Set page configuration
st.set_page_config(
    page_title="Fashion Recommender",
    page_icon="👔",
    layout="wide"
)

# Define user profile class
@dataclass
class UserProfile:
    gender: str
    height: float
    weight: float
    bust_chest: float
    waist: float
    hips: Optional[float] = None
    preferred_colors: List[str] = None
    preferred_styles: List[str] = None
    preferred_seasons: List[str] = None

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """Load and clean the dataset"""
    try:
        df = pd.read_csv(
            file_path,
            encoding='utf-8',
            on_bad_lines='skip'
        )
        
        # Keep only required columns
        required_columns = [
            'id', 'gender', 'masterCategory', 'subCategory', 'articleType',
            'baseColour', 'season', 'year', 'usage', 'productDisplayName'
        ]
        
        df = df[required_columns]
        
        # Clean data
        df = df.dropna()
        df['id'] = df['id'].astype(int)
        df['year'] = df['year'].astype(int)
        
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

class BodyTypeAnalyzer:
    """Analyze and determine body type based on measurements"""
    
    def __init__(self):
        self.body_types = {
            'hourglass': {
                'description': 'Balanced bust and hips with defined waist',
                'recommended': ['Fitted', 'Wrap', 'Belt at waist'],
                'avoid': ['Boxy', 'Oversized']
            },
            'pear': {
                'description': 'Hips larger than bust',
                'recommended': ['A-line', 'Boot cut', 'Empire waist'],
                'avoid': ['Skinny fit', 'Pencil cut']
            },
            'apple': {
                'description': 'Fuller midsection',
                'recommended': ['Empire line', 'A-line', 'V-neck'],
                'avoid': ['Tight waist', 'Belt']
            },
            'rectangle': {
                'description': 'Straight figure with little curve',
                'recommended': ['Ruffles', 'Layers', 'Peplum'],
                'avoid': ['Straight cut', 'Shapeless']
            }
        }
    
    def calculate_bmi(self, height: float, weight: float) -> float:
        """Calculate BMI"""
        return weight / ((height/100) ** 2)
    
    def determine_body_type(self, profile: UserProfile) -> Dict:
        """Determine body type based on measurements"""
        bmi = self.calculate_bmi(profile.height, profile.weight)
        
        # Calculate body ratios
        waist_hip_ratio = profile.waist / profile.hips if profile.hips else None
        bust_hip_diff = abs(profile.bust_chest - profile.hips) if profile.hips else None
        waist_hip_diff = profile.hips - profile.waist if profile.hips else None
        
        # Determine body type
        if bust_hip_diff and waist_hip_diff:
            if bust_hip_diff < 3.6 and waist_hip_diff >= 22.5:
                body_type = 'hourglass'
            elif profile.hips > profile.bust_chest + 5:
                body_type = 'pear'
            elif profile.bust_chest > profile.hips + 5:
                body_type = 'apple'
            else:
                body_type = 'rectangle'
        else:
            body_type = 'rectangle'  # default if measurements are incomplete
        
        return {
            'body_type': body_type,
            'bmi': bmi,
            'characteristics': self.body_types[body_type]
        }

class FashionRecommender:
    """Main recommendation system"""
    
    def __init__(self, data_path: str, images_path: str):
        """Initialize the recommendation system"""
        self.df = load_and_clean_data(data_path)
        self.images_path = Path(images_path)
        self.body_analyzer = BodyTypeAnalyzer()
    
    def get_recommendations(self, profile: UserProfile, n_recommendations: int = 6) -> pd.DataFrame:
        """Get personalized fashion recommendations"""
        # Start with gender filter
        recommendations = self.df[self.df['gender'] == profile.gender].copy()
        
        # Get body type and apply relevant filters
        body_analysis = self.body_analyzer.determine_body_type(profile)
        
        # Apply filters based on user preferences
        if profile.preferred_colors:
            recommendations = recommendations[
                recommendations['baseColour'].isin(profile.preferred_colors)
            ]
        
        if profile.preferred_styles:
            recommendations = recommendations[
                recommendations['usage'].isin(profile.preferred_styles)
            ]
        
        if profile.preferred_seasons:
            recommendations = recommendations[
                recommendations['season'].isin(profile.preferred_seasons)
            ]
        
        # Sort by year (prefer recent items) and select top n
        recommendations = recommendations.sort_values('year', ascending=False)
        return recommendations.head(n_recommendations)
    
    def get_image_path(self, product_id: int) -> str:
        """Get full path to product image"""
        return str(self.images_path / f"{product_id}.jpg")

def main():
    """Main application function"""
    
    st.title("👔 Fashion Recommendation System")
    st.write("Get personalized fashion recommendations based on your body type and preferences!")
    
    # Initialize recommender
    if 'recommender' not in st.session_state:
        try:
            st.session_state.recommender = FashionRecommender(
                data_path='styles.csv',
                images_path='images'
            )
        except Exception as e:
            st.error(f"Error initializing recommender: {str(e)}")
            return
    
    # Sidebar for user inputs
    st.sidebar.header("Your Information")
    
    # Basic Information
    gender = st.sidebar.selectbox(
        "Gender",
        options=["Men", "Women", "Boys", "Girls"]
    )
    
    # Body Measurements
    st.sidebar.subheader("Body Measurements")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        height = st.number_input("Height (cm)", 100.0, 250.0, 170.0)
        bust_chest = st.number_input("Bust/Chest (cm)", 50.0, 150.0, 90.0)
        hips = st.number_input("Hips (cm)", 50.0, 150.0, 95.0)
    
    with col2:
        weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0)
        waist = st.number_input("Waist (cm)", 50.0, 150.0, 75.0)
    
    # Style Preferences
    st.sidebar.subheader("Style Preferences")
    
    # Get unique values from the dataset
    colors = sorted(st.session_state.recommender.df['baseColour'].unique())
    styles = sorted(st.session_state.recommender.df['usage'].unique())
    seasons = sorted(st.session_state.recommender.df['season'].unique())
    
    preferred_colors = st.sidebar.multiselect("Preferred Colors", colors)
    preferred_styles = st.sidebar.multiselect("Preferred Styles", styles)
    preferred_seasons = st.sidebar.multiselect("Preferred Seasons", seasons)
    
    # Create user profile
    user_profile = UserProfile(
        gender=gender,
        height=height,
        weight=weight,
        bust_chest=bust_chest,
        waist=waist,
        hips=hips,
        preferred_colors=preferred_colors,
        preferred_styles=preferred_styles,
        preferred_seasons=preferred_seasons
    )
    
    # Get recommendations button
    if st.sidebar.button("Get Recommendations"):
        with st.spinner("Analyzing your body type and finding recommendations..."):
            # Get body type analysis
            body_analysis = st.session_state.recommender.body_analyzer.determine_body_type(user_profile)
            
            # Display body type information
            st.header("Your Body Type Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"Body Type: {body_analysis['body_type'].title()}")
                st.write(body_analysis['characteristics']['description'])
                st.write(f"BMI: {body_analysis['bmi']:.1f}")
            
            with col2:
                st.subheader("Style Recommendations")
                st.write("👍 Recommended Styles:")
                st.write(", ".join(body_analysis['characteristics']['recommended']))
                st.write("👎 Styles to Avoid:")
                st.write(", ".join(body_analysis['characteristics']['avoid']))
            
            # Get and display recommendations
            st.header("Recommended Products")
            recommendations = st.session_state.recommender.get_recommendations(user_profile)
            
            # Display recommendations in a grid
            cols = st.columns(3)
            for idx, (_, row) in enumerate(recommendations.iterrows()):
                with cols[idx % 3]:
                    try:
                        image_path = st.session_state.recommender.get_image_path(row['id'])
                        if os.path.exists(image_path):
                            image = Image.open(image_path)
                            st.image(image, caption=row['productDisplayName'])
                        else:
                            st.write("Image not available")
                    except Exception as e:
                        st.write("Error loading image")
                    
                    st.write(f"Type: {row['articleType']}")
                    st.write(f"Color: {row['baseColour']}")
                    st.write(f"Season: {row['season']}")
                    st.write("---")

if __name__ == "__main__":
    main()