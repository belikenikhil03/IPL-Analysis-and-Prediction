import streamlit as st
import pickle
import pandas as pd

# Set page config
st.set_page_config(page_title="IPL Win Predictor", layout="wide")

# Custom CSS with blurred background image
st.markdown("""
<style>
    .stApp {
        background-image: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)), url("https://wallpapercave.com/wp/wp4059913.jpg");
        background-size: cover;
        background-position: center;
    }
    .container {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 10px;
    }
    .big-font {
        font-size: 36px !important;
        font-weight: bold;
        color: #FFD700;
        text-shadow: 2px 2px 4px #000000;
    }
    .result-font {
        font-size: 24px !important;
        font-weight: bold;
        color: #FFFFFF;
        text-shadow: 2px 2px 4px #000000;
    }
    .stSelectbox>div>div>div {
        background-color: rgba(255, 255, 255, 0.8);
    }
    .stNumberInput>div>div>input {
        background-color: rgba(255, 255, 255, 0.8);
    }
    .stButton>button {
        background-color: #FFD700;
        color: #000000;
        font-weight: bold;
    }
    .white-text {
        color: #FFFFFF;
        text-shadow: 1px 1px 2px #000000;
    }
</style>
""", unsafe_allow_html=True)

# Load data and model
teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore', 'Kolkata Knight Riders',
         'Kings XI Punjab', 'Chennai Super Kings', 'Rajasthan Royals', 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi', 'Chandigarh', 'Jaipur', 'Chennai',
          'Cape Town', 'Port Elizabeth', 'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala', 'Visakhapatnam', 'Pune', 'Raipur',
          'Ranchi', 'Abu Dhabi', 'Sharjah', 'Mohali', 'Bengaluru']

pipe = pickle.load(open('pipe.pkl', 'rb'))

# Main container
with st.container():
    st.markdown('<div class="container">', unsafe_allow_html=True)

    # Title
    st.markdown("<p class='big-font'>IPL Win Predictor 🏏</p>", unsafe_allow_html=True)

    # Input fields
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<p class='white-text'>Batting Team</p>", unsafe_allow_html=True)
        batting_team = st.selectbox('', sorted(teams), key='batting_team')
        st.markdown("<p class='white-text'>Target</p>", unsafe_allow_html=True)
        target = st.number_input('', min_value=0, step=1, key='target')

    with col2:
        st.markdown("<p class='white-text'>Bowling Team</p>", unsafe_allow_html=True)
        bowling_team = st.selectbox('', sorted(teams), key='bowling_team')
        st.markdown("<p class='white-text'>Current Score</p>", unsafe_allow_html=True)
        score = st.number_input('', min_value=0, step=1, key='score')

    with col3:
        st.markdown("<p class='white-text'>Overs Completed</p>", unsafe_allow_html=True)
        overs = st.number_input('', min_value=0.0, max_value=20.0, step=0.1, key='overs')
        st.markdown("<p class='white-text'>Wickets Out</p>", unsafe_allow_html=True)
        wickets = st.number_input('', min_value=0, max_value=10, step=1, key='wickets')

    st.markdown("<p class='white-text'>Host City</p>", unsafe_allow_html=True)
    selected_city = st.selectbox('', sorted(cities), key='city')

    # Prediction button
    if st.button('Predict Win Probability', key='predict'):
        runs_left = target - score
        balls_left = 120 - (overs * 6)
        wickets_left = 10 - wickets
        crr = score / overs if overs > 0 else 0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0
        
        input_df = pd.DataFrame({
            'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': [selected_city],
            'runs_left': [runs_left], 'balls_left': [balls_left], 'wickets': [wickets_left],
            'total_runs_x': [target], 'crr': [crr], 'rrr': [rrr]
        })
        
        result = pipe.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]
        
        # Display result
        st.markdown("<p class='result-font'>Win Probability:</p>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<p class='result-font'>{batting_team}: {round(win*100)}%</p>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<p class='result-font'>{bowling_team}: {round(loss*100)}%</p>", unsafe_allow_html=True)

    # Footer
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p class='white-text'>Made with ❤️ for cricket fans</p>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)