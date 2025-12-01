from database import db, Clothing
from backend.weather import get_current_weather
from sqlalchemy import func

COLD_WEAR = ["Manteau", "Veste", "Pull", "Chemise"]
WARM_WEAR = ["T-shirt", "Short", "Jupe", "Robe"]
FOOTWEAR_WATERPROOF = ["Chaussures"]

def get_available_categories():
    # SELECT category, COUNT(category) FROM clothing GROUP BY category
    available = db.session.query(
        Clothing.category, 
        func.count(Clothing.category)
    ).group_by(Clothing.category).all()
    
    # Returns { 'T-shirt': 5, 'Jean': 2, ... }
    return {cat: count for cat, count in available}


def generate_outfit(city="Paris"):
    weather = get_current_weather(city)
    available_categories = get_available_categories()
    
    outfit = {
        'weather': weather,
        'suggestions': [],
        'comment': f"Prévisions à {city} : {weather.temperature}°C, {weather.description}."
    }

    is_cold = weather.temperature < 10
    is_warm = weather.temperature > 25
    is_rainy = weather.condition_key in ["rain", "thunderstorm"]
    
    suggested_top_category = ""
    suggested_bottom_category = ""
    
    if is_cold:
        outfit['comment'] += "🥶 Il fait froid !"
        suggested_top_category = "Manteau"
        suggested_bottom_category = "Pantalon"
    elif is_warm:
        outfit['comment'] += "😎 Le soleil est au rendez-vous, privilégie le léger."
        suggested_top_category = "T-shirt"
        suggested_bottom_category = "Short"
    elif is_rainy :
        outfit['comment'] += "Attention il pleut."
        suggested_top_category = "Manteau"
        suggested_bottom_category = "Jean"
    
    if suggested_top_category in available_categories:
        top_item = Clothing.query.filter_by(category=suggested_top_category).first()
        outfit['suggestions'].append({
                'item': top_item,
                'advice': f"Haut : {top_item.category}. (Tu as {available_categories.get(top_item.category)} disponibles)"
            })
    else:
        outfit['suggestions'].append({
            'item': None,
            'advice': f"Haut : Nous n'avons pas trouvé de {suggested_top_category} dans ta garde-robe. Pense à en ajouter !"
        })
    
    if suggested_bottom_category in available_categories:
        bottom_item = Clothing.query.filter_by(category=suggested_bottom_category).first()
        if bottom_item:
            outfit['suggestions'].append({
                'item': bottom_item,
                'advice': f"Bas : {bottom_item.category}."
            })
    
    # FOOTWEAR (Adaptation Pluie)
    if is_rainy:
        outfit['comment'] += " Pense à tes chaussures imperméables !"
        if "Chaussures" in available_categories:
             # Idéalement, ici on filtrerait les chaussures de type 'Bottes' ou 'Baskets'
            foot_item = Clothing.query.filter_by(category="Chaussures").first() 
            if foot_item:
                outfit['suggestions'].append({
                    'item': foot_item,
                    'advice': f"Chaussures : Tes {foot_item.category} sont recommandées contre la pluie."
                })
        else:
            outfit['suggestions'].append({
                'item': None,
                'advice': "Ajoutez des chaussures à votre garde-robe pour une meilleure suggestion."
            })

    return outfit