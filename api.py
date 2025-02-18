import requests
import pandas as pd
import json


uri = "https://api.football-data.org/v4/matches"
headers = {'X-Auth-Token': ''}
response = requests.get(uri, headers=headers)
data = response.json()
with open('data.json', 'w') as f:
   json.dump(data, f)

with open("data.json", "r") as file:
   data = json.load(file)

matches = data['matches']
structured_data = []

for match in matches: 
   structured_data.append({
      "competition": match["competition"]["name"],
      "match_date": match["utcDate"],
      "home_team": match["homeTeam"]["name"],
      "away_team": match["awayTeam"]["name"],
      "home_score": match["score"]["fullTime"].get("home", None),
      "away_score": match["score"]["fullTime"].get("away", None),
      "match_status": match["status"]
   })

df = pd.DataFrame(structured_data)
print(df.head())
df.info()
df.isnull().sum()
df.nunique()
df.describe()