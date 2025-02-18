import requests
import pandas as pd
import json
import seaborn as seaborn
import matplotlib.pyplot as plt
from datetime import datetime


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

for team in teams:
   team_name = team["team"]["name"]
   team_stats[team_name] = {
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
   match_date = datetime.strptime(match["utcDate"], '%Y-%m-%dT%H:%M:%SZ')
   goal_diff = (home_score - away_score)


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


df2 = pd.DataFrame(structured_data)
print(df2.head())
df2.info()
df2.isnull().sum()
df2.nunique()
df2.describe()

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

