import requests
from uagents import Model, Field

API_KEY = "YOUR_ALLSPORTS_API_KEY"
BASE_URL = "https://apiv2.allsportsapi.com/football/"

class FootballTeamRequest(Model):
    team_name: str

class FootballTeamResponse(Model):
    results: str

async def get_team_info(team_name: str) -> str:
    """
    Fetch team information from AllSportsAPI and return as plain text
    """
    try:
        params = {
            "met": "Teams",
            "teamName": team_name,
            "APIkey": API_KEY
        }

        response = requests.get(BASE_URL, params=params)
        data = response.json()

        if data.get("success") == 1 and data.get("result"):
            team_info = data["result"][0]
            result = f"\nTeam Name: {team_info['team_name']}\n"
            result += f"Team Logo: {team_info['team_logo']}\n\n"
            result += "Players:\n"
            
            for player in team_info.get("players", []):
                result += f"- Name: {player['player_name']}\n"
                result += f"  Type: {player['player_type']}\n"
                result += f"  Image: {player['player_image']}\n\n"
            
            return result
        else:
            return "Team not found or invalid API key."
            
    except Exception as e:
        return f"Error fetching team information: {str(e)}"