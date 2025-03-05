import requests
import pandas as pd
import json
import seaborn as seaborn
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import spearmanr, pearsonr
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



uri = "https://api.football-data.org/v4/competitions/PL/"
finishedMatches = "matches?status=FINISHED"
standings = "standings"


headers = {'X-Auth-Token': ''}
response = requests.get(uri + finishedMatches, headers=headers)
data = response.json()
with open('dataPLMatches.json', 'w') as f:
   json.dump(data, f)

with open("dataPLMatches.json", "r") as file:
   data = json.load(file)

responseTeam = requests.get(uri + standings, headers=headers)
dataTeam = responseTeam.json()

with open('dataPLStandings.json', 'w') as f:
   json.dump(dataTeam, f)

with open("dataPLStandings.json", "r") as file:
   dataTeam = json.load(file)

matches = data['matches']
teams = dataTeam['standings'][0]['table']
structured_data = []
team_stats = {}

team_name_mapping = {
    "AFC Bournemouth": 1,
    "Arsenal FC": 2,
    "Aston Villa FC": 3,
    "Brentford FC": 4,
    "Brighton & Hove Albion FC": 5,
    "Chelsea FC": 6,
    "Crystal Palace FC": 7,
    "Everton FC": 8,
    "Fulham FC": 9,
    "Ipswich Town FC": 10,
    "Leicester City FC": 11,
    "Liverpool FC": 12,
    "Manchester City FC": 13,
    "Manchester United FC": 14,
    "Newcastle United FC": 15,
    "Nottingham Forest FC": 16,
    "Southampton FC": 17,
    "Tottenham Hotspur FC": 18,
    "West Ham United FC": 19,
    "Wolverhampton Wanderers FC": 20
}


for team in teams:
   team_name = team["team"]["name"]
   team_id = team_name_mapping.get(team_name, -1)
   team_stats[team_name] = {
      "teamId": team_id,
      "position": team["position"],
      "playedGames": team["playedGames"],
      "won": team["won"],
      "draw": team["draw"],
      "lost": team["lost"],
      "goalsFor": team["goalsFor"],
      "goalsAgainst": team["goalsAgainst"],
      "goalDifference": team["goalDifference"],
      "points": team["points"]
   }

for match in matches: 
   home_score = match["score"]["fullTime"].get("home", None)
   away_score = match["score"]["fullTime"].get("away", None)
   home_team = match["homeTeam"]["name"]
   away_team = match["awayTeam"]["name"]
   home_team_id = team_name_mapping.get(home_team, -1)
   away_team_id = team_name_mapping.get(away_team, -1)
   match_date = datetime.strptime(match["utcDate"], '%Y-%m-%dT%H:%M:%SZ')
   goal_diff = None if home_score is None or away_score is None else (home_score - away_score)


   if home_score is None or away_score is None:
      result = "Unknown"
   elif home_score > away_score:
      result = "Home win"
   elif home_score < away_score:
      result = "Away win"
   else:
      result = "Draw"

   structured_data.append({
      "competition": match["competition"]["name"],
      "match_date": match_date,
      "home_team": home_team,
      "away_team": away_team,
      "home_score": home_score,
      "away_score": away_score,
      "result": result,
      "match_status": match["status"],
      "goal_difference": goal_diff
   })


df = pd.DataFrame.from_dict(team_stats, orient="index").reset_index()
df.rename(columns={"index": "team"}, inplace=True)
print(df.head())
df.info()
df.isnull().sum()
df.nunique()
df.describe()


def calculate_form(team, match_date):
   past_matches = df2[
   ((df2['home_team'] == team) | (df2['away_team'] == team))
   & (df2['match_date'] < match_date)
   ].sort_values(by='match_date', ascending=False).head(5)

   points = 0
   total_points = 15
   for _, row in past_matches.iterrows():
      if row['home_team'] == team:
         if row['result'] == "Home win":
            points += 3
         elif row['result'] == "Draw":
            points += 1
      elif row['away_team'] == team:
         if row['result'] == "Away win":
            points += 3
         elif row['result'] == "Draw":
            points += 1
   percentage_of_points = (points / total_points) * 100
   return percentage_of_points.__round__(2)

def calculate_goal_difference(team, match_date):
   past_matches = df2[
   ((df2['home_team'] == team) | (df2['away_team'] == team))
   & (df2['match_date'] < match_date)
   ].sort_values(by='match_date', ascending=False).head(5)
   goal_difference = 0

   for _, row in past_matches.iterrows():
      if row['home_team'] == team:
         goal_difference += (row['home_score'] - row['away_score'])
      elif row['away_team'] == team:
         goal_difference += (row['away_score'] - row['home_score'])
   return goal_difference


df2 = pd.DataFrame(structured_data)

df2 = df2.sort_values(by='match_date', ascending=False)
print(df2.head())
df2.info()
df2.isnull().sum()
df2.nunique()
df2.describe()

df2['result_numeric'] = df2['result'].map({"Home win" : 1, "Draw" : 0, "Away win" : -1})
df2["home_team_form"] = df2.apply(lambda row: calculate_form(row["home_team"], row["match_date"]), axis=1)
df2["away_team_form"] = df2.apply(lambda row: calculate_form(row["away_team"], row["match_date"]), axis=1)
df2["home_team_goal_difference"] = df2.apply(lambda row: calculate_goal_difference(row["home_team"], row["match_date"]), axis=1)
df2["away_team_goal_difference"] = df2.apply(lambda row: calculate_goal_difference(row["away_team"], row["match_date"]), axis=1)
df2["home_team_strength"] = df2["home_team_form"] + df2["home_team_goal_difference"]
df2["away_team_strength"] = df2["away_team_form"] + df2["away_team_goal_difference"]
df2["goal_diff_delta"] = df2["home_team_goal_difference"] - df2["away_team_goal_difference"]

corr, p_value = spearmanr(df2["home_team_strength"], df2["result_numeric"])
print(f"Spearsman's correlation: {corr:.3f}, p_value: {p_value:.3f}")
pearson_corr, p_value = pearsonr(df2["home_team_strength"], df2["result_numeric"])
print(f"Pearson's correlation: {pearson_corr:.3f}, p_value: {p_value:.3f}")

X = df2[["goal_diff_delta", "home_team_strength"]]
y = df2["result_numeric"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(class_weight="balanced")
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Classification Report: ", classification_report(y_test, y_pred))
print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))

#seaborn.boxplot(x=df2["goal_diff_delta"], y=df2["result_numeric"])
#plt.title("Goal Difference Delta vs. Match Result")
#plt.xlabel("Goal Difference Delta (Home - Away)")
#plt.ylabel("Match Result")
#plt.savefig("Boxplot of Goal Difference Delta and Match Result")
#plt.show()



#seaborn.boxplot(x=df2["home_team_strength"], y=df2["result_numeric"])
#plt.title("Correlation between Home Team Strength and Match Result")
#plt.xlabel("Home Team Strength (Form + Goal Difference)")
#plt.ylabel("Match Result")
#plt.savefig("Boxplot of Home Team Strength and Match Result")
#plt.show()

#seaborn.boxplot(x=df2["home_team_goal_difference"], y=df2["result_numeric"])
#plt.title("Correlation between Home Team Goal Difference and Match Result")
#plt.xlabel("Home Team Goal Difference (Last 5 matches)")
#plt.ylabel("Match Result")
#plt.savefig("Boxplot of Home Team Goal Difference and Match Result")
#plt.show()

#seaborn.boxplot(x=df2["away_team_goal_difference"], y=df2["result_numeric"])
#plt.title("Correlation between Away Team Goal Difference and Match Result")
#plt.xlabel("Away Team Goal Difference (Last 5 matches)")
#plt.ylabel("Match Result")
#plt.savefig("Boxplot of Away Team Goal Difference and Match Result")
#plt.show()

#seaborn.boxplot(x=df2["home_team_form"], y=df2["result_numeric"])
#plt.title("Correlation between Home Team Form and Match Result")
#plt.xlabel("Home Team Form(%)")
#plt.ylabel("Match Result")
#plt.savefig("Boxplot of Home Team Form and Match Result")
#plt.show()





# seaborn.histplot(df['result'])
# plt.title('Histogram of Match Results')
# plt.xlabel('Result')
# plt.ylabel('Frequency')
# plt.savefig('Histogram of Match Results')
# plt.show()

# seaborn.boxplot(x=df['result'], y=df['home_score'])
# plt.title('Boxplot of Home Scores by Match Result')
# plt.xlabel('Result')
# plt.ylabel('Home Score')
# plt.savefig('Boxplot of Home Scores')
# plt.show()

# seaborn.boxplot(x=df['result'], y=df['away_score'])
# plt.title('Boxplot of Away Scores by Match Result')
# plt.xlabel('Result')
# plt.ylabel('Away Score')
# plt.savefig('Boxplot of Away Scores')
# plt.show()

