from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model
pipe = pickle.load(open(r"E:\Machine Learning\Projects\cricket_score_predictor-main\pipe.pkl", 'rb'))

teams = [
    'Australia', 'India', 'Bangladesh', 'New Zealand', 'South Africa', 
    'England', 'West Indies', 'Afghanistan', 'Pakistan', 'Sri Lanka'
]

cities = [
    'Colombo', 'Mirpur', 'Johannesburg', 'Dubai', 'Auckland', 'Cape Town', 
    'London', 'Pallekele', 'Barbados', 'Sydney', 'Melbourne', 'Durban', 
    'St Lucia', 'Wellington', 'Lauderhill', 'Hamilton', 'Centurion', 
    'Manchester', 'Abu Dhabi', 'Mumbai', 'Nottingham', 'Southampton', 
    'Mount Maunganui', 'Chittagong', 'Kolkata', 'Lahore', 'Delhi', 
    'Nagpur', 'Chandigarh', 'Adelaide', 'Bangalore', 'St Kitts', 'Cardiff', 
    'Christchurch', 'Trinidad'
]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        batting_team = request.form['batting_team']
        bowling_team = request.form['bowling_team']
        city = request.form['city']
        current_score = int(request.form['current_score'])
        overs = float(request.form['overs'])
        wickets = int(request.form['wickets'])
        last_five = int(request.form['last_five'])


        balls_left = 120 - (overs * 6)
        wicket_left = 10 - wickets
        current_run_rate = current_score / overs

        input_df = pd.DataFrame(
            {'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': [city], 
             'current_score': [current_score], 'balls_left': [balls_left], 
             'wicket_left': [wicket_left], 'current_run_rate': [current_run_rate], 'last_five': [last_five]}
        )

        result = pipe.predict(input_df)
        prediction = int(result[0])

    bg_image_url = '/static/images/acricket.jpg'
    return render_template('index.html', teams=sorted(teams), cities=sorted(cities), prediction=prediction, bg_image_url=bg_image_url)

if __name__ == '__main__':
    app.run(debug=True)
