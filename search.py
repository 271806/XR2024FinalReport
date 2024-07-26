import os
import io
from google.cloud import vision
import requests


def analyze_image(file_path):
    """Analyzes an image file for labels and text."""
    client = vision.ImageAnnotatorClient()

    with io.open(file_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    # Perform label detection
    label_response = client.label_detection(image=image)
    labels = label_response.label_annotations

    # Perform text detection
    text_response = client.text_detection(image=image)
    texts = text_response.text_annotations

    return labels, texts

def get_nutrition_info(query):
    """Queries the USDA API for nutritional information."""
    api_key = "YOUR_USDA_API_KEY"  # Replace with your actual USDA API key
    base_url = "https://api.nal.usda.gov/fdc/v1/foods/search"
    
    params = {
        "api_key": api_key,
        "query": query,
        "dataType": ["Survey (FNDDS)"],
        "pageSize": 1
    }
    
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['foods']:
            return data['foods'][0]
    return None

def main():
    crops_dir = "output/crops"  # Directory containing cropped images
    results = {}

    for filename in os.listdir(crops_dir):
        if filename.endswith(".png"):
            file_path = os.path.join(crops_dir, filename)
            print(f"Analyzing image {filename}...")
            
            labels, texts = analyze_image(file_path)
            
            # Use the first label as our snack identification
            if labels:
                snack_name = labels[0].description
                print(f"Identified as: {snack_name}")
                
                # Get nutrition info
                nutrition = get_nutrition_info(snack_name)
                if nutrition:
                    results[filename] = {
                        'snack': snack_name,
                        'calories': nutrition.get('foodNutrients', [{}])[0].get('value', 'N/A'),
                        'protein': nutrition.get('foodNutrients', [{}])[1].get('value', 'N/A'),
                        'fat': nutrition.get('foodNutrients', [{}])[2].get('value', 'N/A'),
                        'carbs': nutrition.get('foodNutrients', [{}])[3].get('value', 'N/A')
                    }
                else:
                    results[filename] = {'snack': snack_name, 'nutrition': 'Not found'}
            else:
                results[filename] = {'snack': 'Unidentified', 'nutrition': 'N/A'}

    # Print results
    for filename, result in results.items():
        print(f"{filename}: {result['snack']}")
        if 'calories' in result:
            print(f"  Calories: {result['calories']}")
            print(f"  Protein: {result['protein']}g")
            print(f"  Fat: {result['fat']}g")
            print(f"  Carbs: {result['carbs']}g")
        else:
            print(f"  Nutrition: {result['nutrition']}")

if __name__ == "__main__":
    main()