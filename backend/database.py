class Clothing:
    def __init__(self, image_path, category = None):
        self.category = category
        self.image_path = image_path
    def __repr__(self): #used with print command
        return f"{self.category}"
import random
