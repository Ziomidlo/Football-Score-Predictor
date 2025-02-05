import requests
import pandas as pd
import json


uri = "https://api.football-data.org/v4/matches"
headers = {'X-Auth-Token': '26d39f32a17044b7a972763541e4a083'}
response = requests.get(uri, headers=headers)
data = response.json()
with open('data.json', 'w') as f:
   json.dump(data, f)

df = pd.read_json("matches_28_01_2025.json")
print(df)
#for match in data['matches']:
   # print(f"Match: {match['id']}, Home Team: {match['homeTeam']['name']}, Away Team: {match['awayTeam']['name']}")

#plDataSet = requests.get("https://api.football-data.org/v4/competitions/PL/matches", headers=headers)

#plData = pd.DataFrame(plDataSet)
#print(plData)


#GET https://api.football-data.org/v4/competitions/2021/matches
#GET https://api.football-data.org/v4/matches