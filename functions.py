import os
from uagents import Model, Field
import pandas as pd
import kagglehub
from google import genai
from google.genai import types
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import math


client = genai.Client(api_key="AIzaSyCaySv74dP_XlHB_IfFS8tCQrSzfJ-a0Wg")

prediction = ""
class IplMatchRequest(Model):
    team_name_01: str
    team_name_02: str
    date: str

# Download dataset
async def getResponse(match_request):

    team1 = match_request.team_name_01
    team2 = match_request.team_name_02
    year  = match_request.date  # or whatever format you want

    path = kagglehub.dataset_download("mohammadzamakhan/ipl-players-statistics")
    print("Dataset downloaded to:", path)

    # List files
    files = os.listdir(path)
    print("Files in dataset:", files)

    # Use the actual file
    file_path = os.path.join(path, "IPL Player Stat.csv")

    # Load CSV
    df = pd.read_csv(file_path)
    df2 = pd.read_csv("sim_100.csv")

    l = df2.iloc[2]
    individual_test_list = l.tolist()[0:-1]

    # Filter only the relevant columns
    df_filtered = df[[
        'player', 'batting_avg', 'batting_strike_rate',
        'bowling_economy', 'bowling_avg', 'bowling_strike_rate'
    ]].copy()

    print(df_filtered.iloc[1])

    # Rename 'player' to 'name'
    df_filtered.rename(columns={'player': 'name'}, inplace=True)

    # Add 'rookie' column
    # True if any of the stats are missing (rookie), False if all stats exist
    df_filtered['rookie'] = ~df_filtered.notna().all(axis=1)

    # Optional: convert to a numpy array
    arr = df_filtered.to_numpy()

    list_avg = [0, 0, 0, 0, 0]

    ct1 = 0
    for it in arr:
        ct2 = 0
        for i in it[1:-1]:
            if (ct2 == 3 or ct2 == 4):
                if (math.floor(i / 10000) > 0):
                    list_avg[ct2] = list_avg[ct2] + (i / 10000)
                    arr[ct1][ct2 + 1] = (i / 10000)
                else:
                    list_avg[ct2] = list_avg[ct2] + i
                ct2 = ct2 + 1
            else:
                list_avg[ct2] = list_avg[ct2] + i
                ct2 = ct2 + 1
        ct1 = ct1 + 1
            

    print(arr[0])

    ct = 0
    for i in list_avg:
        list_avg[ct] = i / len(arr)
        ct = ct + 1

    print(list_avg)

    # Get the player names
    def get_player_names(arr):
        names = []
        for player in arr:
            names.append(player[0])
        return names

    # Get the stats
    def get_stats(all_players):
        a = []
        for player in all_players:
            a.append(player[1:-1])  # everything except the name
        return a  # if player not found

    names_l = get_player_names(arr)
    stats = get_stats(arr)



    # Plan
    # -- Add filler for rookies (possibly)
    # -- 
    # -- Train the model ( -- SR does it -- )



    over_arch_list = []
    def createTeamLists(individual_test):
        sum = 0
        counter = 0
        list_team1 = []
        list_team2 = []

        for s in names_l:

            counter2 = 0
            for s2 in individual_test:
                first1 = s.split()[0]
                first2 = s.split()[-1]
                last1 = s2.split()[0]
                last2 = s2.split()[-1]
                if (first1[0] == last1[0] and first2 == last2):
                    #add the 5 special traits
                    if (counter2 < 11):
                        list_team1.append(arr[counter][1:-1])
                    else:
                        list_team2.append(arr[counter][1:-1])
                    sum = sum + 1
                counter2 = counter2 + 1
            counter = counter + 1
        over_arch_list.append(list_team1)
        over_arch_list.append(list_team2)

        # print(over_arch_list)

    # ground_t_l = l.tolist()[-1]

    # print(ground_t_l)

    # Example function to simulate input arrays and labels
    # print("Total Rows: " + str(len(df2)))

    # print("over arch" + str(len(over_arch_list)))

    # l = df2.iloc[2]
    # individual_test_list = l.tolist()[0:-1]

    numMatches = len(df2)
    def generate_sample_ipl_data(num_matches=numMatches):
        X = []
        y = []
        count1 = 0
        
        for _ in range(num_matches):
            l = df2.iloc[count1]
            individual_test_list = l.tolist()[0:-1]
            createTeamLists(individual_test_list)
        
            # Simulate team 1 and team 2 player stats (11 players × 5 stats)
            team1 = over_arch_list[0]
            team2 = over_arch_list[1]
            print(team1)
            print(team2)
            # Flatten and concatenate teams
            features = np.concatenate([np.array(team1).flatten(), np.array(team2).flatten()])
            X.append(features)
            # Random winner label (0: team1 wins, 1: team2 wins)
            ground_t_l = l.tolist()[-1]
            y.append(ground_t_l)

            count1 = count1 + 1

            print(count1)
        
        print("X len" + str(len(X)))
        print("X len" + str(len(y)))

        return np.array(X), np.array(y)



    # Generate dataset
    X, y = generate_sample_ipl_data(numMatches)

    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features (important for consistent model training)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Predict on test data
    y_pred = model.predict(X_test_scaled)

    # Evaluate the model
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Download dataset
    path = kagglehub.dataset_download("mohammadzamakhan/ipl-players-statistics")
    print("Dataset downloaded to:", path)

    # List files
    files = os.listdir(path)
    print("Files in dataset:", files)

    # Use the actual file
    file_path = os.path.join(path, "IPL Player Stat.csv")

    # Load CSV
    df = pd.read_csv(file_path)

    # Filter only the relevant columns
    df_filtered = df[[
        'player', 'batting_avg', 'batting_strike_rate',
        'bowling_economy', 'bowling_avg', 'bowling_strike_rate'
    ]].copy()

    # Rename 'player' to 'name'
    df_filtered.rename(columns={'player': 'name'}, inplace=True)

    # Add 'rookie' column
    # True if any of the stats are missing (rookie), False if all stats exist
    df_filtered['rookie'] = ~df_filtered.notna().all(axis=1)

    # Optional: convert to a numpy array
    arr = df_filtered.to_numpy()

    # Print the resulting array or DataFrame
    def get_player_names(arr):
        names = []
        for player in arr:
            names.append(player[0])
        return names

    arr2 = get_player_names(arr)

    # for name in arr2:
    #     print(name)

    #only print the names that are relevant to this match
    prompt = f"""
    Return a valid JSON array of the 11 players for the {year} {team1} squad that most matches the players in the list "arr2".
    Only include player names.
    Can you reference data from the present please

    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json" # Disables thinking
        ),
    )

    text = response.text
    print(text)

    try:
        team1_list = json.loads(text)
        print(team1_list)
    except json.JSONDecodeError:
        print("Failed to parse response as JSON")
        print(response.text)

    prompt = f"""
    Return a valid JSON array of the 11 players for the {year} {team2} squad that most matches the players in the list "arr2".

    Example output: ["Player 1", "Player 2", ..., "Player 11"]


    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0), # Disables thinking
            response_mime_type="application/json"
        ),
    )

    text = response.text

    try:
        team2_list = json.loads(text)
        print(team2_list)
    except json.JSONDecodeError:
        print("Failed to parse response as JSON")
        print(response.text)

    # print(team1_list['rookie'])
    # for i in range(0, 11):
    #going through the list of names kumar provides, set the name

    testArr = []
    for i in range(0, len(arr2)):
        for j in range(0, len(team1_list)):
            team1_text = team1_list[j].strip().split()[-1]
            arr2_text = arr2[i].strip().split()[-1]
            if (team1_list[j][0] == arr2[i][0] and team1_text == arr2_text ):
            
                testArr.append(arr[i][1:-1])

    for i in range(0, len(arr2)):
        for j in range(0, len(team2_list)):
            team2_text = team2_list[j].strip().split()[-1]
            arr2_text = arr2[i].strip().split()[-1]
            if (team2_list[j][0] == arr2[i][0] and team2_text == arr2_text ):
                testArr.append(arr[i][1:-1])
    print(testArr)
    print(len(testArr))


    # user_input = np.concatenate(testArr).astype(float)   # replace with actual feature values matching model input shape

    # print(user_input.shape) 
    # # Scale the user input just like training data
    # user_input_scaled = scaler.transform(user_input.reshape(1, -1))

    # # Predict single example
    # prediction = model.predict(user_input_scaled)

    # print("Predicted winner:", prediction[0])


    user_input = np.concatenate(testArr).astype(float)
    expected_features = X.shape[1]  # features the model was trained on
    current_features = user_input.shape[0]

    if current_features < expected_features:
        # Pad with zeros
        padding = np.zeros(expected_features - current_features)
        user_input = np.concatenate([user_input, padding])

    user_input = user_input.reshape(1, -1)
    user_input_scaled = scaler.transform(user_input)

    prediction = model.predict(user_input_scaled)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Test Accuracy: {acc:.2f}")

    print("Predicted winner:", prediction[0])
    winner = ""
    loser = ""
    output = ""
    
    if (prediction[0] == "Team 1"):
        winner = team1
       
    else:
        winner = team2
      

    # prompt = f"""Print the {winner}. Accuracy of model is {acc}"""

    # response = client.models.generate_content(
    #     model="gemini-2.5-flash",
    #     contents=prompt,
    #     config=types.GenerateContentConfig(
    #         thinking_config=types.ThinkingConfig(thinking_budget=0), # Disables thinking
    #     ),
    # )

    # text = response.text
    output = f"The predicted winner is {winner}." 
    return output

def setResponse():
    return prediction[0]


