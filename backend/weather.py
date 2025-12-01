import random
from datetime import datetime

CONDITIONS = {
    "clear": "Soleil",
    "clouds": "Nuageux",
    "rain": "Pluie",
    "snow": "Neige",
    "thunderstorm": "Orages",
    "mist": "Brume"
}

SIMULATED_TEMPS = {
    "winter": (0, 8, "snow"),
    "spring": (8, 18, "clouds"),
    "summer": (20, 32, "clear"),
    "fall": (10, 20, "rain")
}

class Weather:
    def __init__(self, city="Paris", temp=None, condition='clear'):
        self.city = city
        self.date = datetime.now().strftime("%Y-%m-%d")
        if temp is None:
            self._simulate_weather()
        else:
            self.temperature = temp
            self.condition_key = condition
            self.description = CONDITIONS.get(self.condition_key, "Inconnu")

    def _get_season(self):
        month = datetime.now().month
        if 3 <= month <= 5: return "spring"
        if 6 <= month <= 8: return "summer"
        if 9 <= month <= 11: return "fall"
        return "winter"

    def _simulate_weather(self):
        season = self._get_season()
        min_temp, max_temp, default_condition = SIMULATED_TEMPS[season]
        
        self.temperature = random.randint(min_temp, max_temp)
        
        # 20% to get default condition in season
        if random.random() < 0.2:
            self.condition_key = default_condition
        else:
            self.condition_key = random.choice(list(CONDITIONS.keys()))
        
        self.description = CONDITIONS.get(self.condition_key, "Inconnu")

    def __repr__(self):
        return f"Météo {self.city}: {self.temperature}°C, {self.description}"

def get_current_weather(city="Paris"):
    return Weather(city=city)
