import requests
import pandas as pd
import json
import csv 
import seaborn as seaborn
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import spearmanr, pearsonr
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


uri = "https://api.football-data.org/v4/competitions/PL/"
matches = "/matches"
finishedMatches =  matches + "?status=FINISHED"
upcomingMaches = matches + "?status=TIMED"
standings = "standings"
cleanedDataFolder = "cleaned_data/"

#Get Data From API
headers = {'X-Auth-Token': '26d39f32a17044b7a972763541e4a083'}
response = requests.get(uri + finishedMatches , headers=headers)
dataFinished = response.json()

with open('dataPLMatches2024Finished.json', 'w') as file:
   json.dump(dataFinished, file)

with open('dataPLMatches2024Finished.json') as file:
   dataFinished = json.load(file)


responseUpcoming = requests.get(uri + upcomingMaches, headers=headers)
dataUpcoming = responseUpcoming.json()

with open('dataPLMatches2024Upcoming.json', 'w') as file:
   json.dump(dataUpcoming, file)

with open('dataPLMatches2024Upcoming.json') as file:
   dataUpcoming = json.load(file)
  
responseTeam = requests.get(uri + standings, headers=headers)
dataTeam = responseTeam.json()

with open('dataPLStandings.json', 'w') as file:
   json.dump(dataTeam, file)

with open('dataPLStandings.json') as file:
   dataTeam = json.load(file)


#Variables

finishedMatches = dataFinished['matches']
upcomingMatches = dataUpcoming['matches']
teamsStandings = dataTeam['standings'][0]['table']
seasons = pd.read_csv(cleanedDataFolder + 'seasons.csv')
pastMatches = pd.read_csv(cleanedDataFolder + 'past-matches.csv')
pastSeasonsStats = pd.read_csv(cleanedDataFolder + 'season-stats.csv')
teams = pd.read_csv(cleanedDataFolder + 'teams.csv') 

#Data Structuring

structuredCurrentSeasonData = []
structuredLastSeasonData = []
structuredTableData = {}

teamMapping = dict(zip(teams['Team'], teams['Id']))

for team in teamsStandings:
   teamName = team["team"]["name"]
   teamId = teamMapping.get(teamName, -1)
   structuredTableData[teamName] = {
      "team_id" : teamId,
      "position": team["position"],
      "played_games": team["playedGames"],
      "won": team["won"],
      "draw": team["draw"],
      "lost": team["lost"],
      "goals_for": team["goalsFor"],
      "goals_against": team["goalsAgainst"],
      "goal_difference": team["goalDifference"],
      "points": team["points"]
   }


for match in finishedMatches:

   homeScore = match["score"]["fullTime"].get("home", None)
   awayScore = match["score"]["fullTime"].get("away", None)
   homeTeam = match["homeTeam"]["name"]
   awayTeam = match["awayTeam"]["name"]
   homeTeamId = teamMapping.get(homeTeam, -1)
   awayTeamId = teamMapping.get(awayTeam, -1)
   currentHomePossition = structuredTableData[homeTeam]["position"]
   CurrentAwayPossition = structuredTableData[awayTeam]["position"] 
   match_date = datetime.strptime(match["utcDate"], '%Y-%m-%dT%H:%M:%SZ')
   goalDiff = None if homeScore is None or awayScore is None else (homeScore - awayScore)
   season = "2024/2025"
   
   if homeScore is None or awayScore is None:
      result = "Unknown"
   elif homeScore > awayScore:
      result = "Home win"
   elif homeScore < awayScore:
      result = "Away win"
   else:
      result = "Draw"

   structuredCurrentSeasonData.append({
      "season": season,
      "competition": match["competition"]["name"],
      "match_date": match_date,
      "home_team_id" : homeTeamId,
      "home_team": homeTeam,
      "away_team_id" : awayTeamId,
      "away_team": awayTeam,
      "home_score": homeScore,
      "away_score": awayScore,
      "result": result,
      "match_status": match["status"],
      "goal_difference": goalDiff
   })

#Creating Data Frames & Analysis
kaggleDf = pd.read_csv("past-dataPL.csv")
print(kaggleDf.head())
kaggleDf.info()
kaggleDf.isnull().sum()
kaggleDf.nunique()
kaggleDf.describe()


df = pd.DataFrame.from_dict(structuredTableData, orient="index").reset_index()
df.rename(columns={"index": "team"}, inplace=True)
print(df.head())
df.info()
df.isnull().sum()
df.nunique()
df.describe()

df2 = pd.DataFrame(structuredCurrentSeasonData)
df2 = df2.sort_values(by='match_date', ascending=False)
print(df2.head())
df2.info()
df2.isnull().sum()
df2.nunique()
df2.describe()



#Feauture Engineering
def calculate_form(team, match_date):
   pastMatches = df2[
   ((df2['home_team'] == team) | (df2['away_team'] == team))
   & (df2['match_date'] < match_date)
   ].sort_values(by='match_date', ascending=False).head(5)

   points = 0
   total_points = 15
   for _, row in pastMatches.iterrows():
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



#Data for model traning
df2['result_numeric'] = df2['result'].map({"Home win" : 1, "Draw" : 0, "Away win" : -1})
df2["home_team_form"] = df2.apply(lambda row: calculate_form(row["home_team"], row["match_date"]), axis=1)
df2["away_team_form"] = df2.apply(lambda row: calculate_form(row["away_team"], row["match_date"]), axis=1)
df2["home_team_goal_difference"] = df2.apply(lambda row: calculate_goal_difference(row["home_team"], row["match_date"]), axis=1)
df2["away_team_goal_difference"] = df2.apply(lambda row: calculate_goal_difference(row["away_team"], row["match_date"]), axis=1)
df2["home_team_strength"] = df2["home_team_form"] + df2["home_team_goal_difference"]
df2["away_team_strength"] = df2["away_team_form"] + df2["away_team_goal_difference"]
df2["goal_diff_delta"] = df2["home_team_goal_difference"] - df2["away_team_goal_difference"]

#Correlation Analysis
corr, p_value = spearmanr(df2["home_team_strength"], df2["result_numeric"])
print(f"Spearsman's correlation: {corr:.3f}, p_value: {p_value:.3f}")
pearson_corr, p_value = pearsonr(df2["home_team_strength"], df2["result_numeric"])
print(f"Pearson's correlation: {pearson_corr:.3f}, p_value: {p_value:.3f}")

#Model Training
X = df2[["goal_diff_delta", "home_team_strength"]]
y = df2["result_numeric"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model =  LogisticRegression(class_weight="balanced")
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Classification Report: ", classification_report(y_test, y_pred))
print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))


#Visualization

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

