import os
from transformers import BlipProcessor, BlipForConditionalGeneration, ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch
import random

# Path to your dataset root
root_dir = r"E:\dataset\images"  # Changed to the main images directory

print("Loading BLIP model for image captioning...")
try:
    # Using BLIP which is better for detailed captioning
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    print("BLIP model loaded successfully!")
except Exception as e:
    print(f"Error loading BLIP model: {e}")
    print("Trying base model...")
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        print("BLIP base model loaded successfully!")
    except Exception as e2:
        print(f"Error loading base model: {e2}")
        processor = None
        model = None

# Enhanced caption templates based on different categories
caption_templates = {
    'wildlife': [
        "detailed view of {subject} in natural habitat, cinematic lighting with contrast, showcasing vivid colors, natural behavior, and environmental context",
        "close-up of {subject} with intricate details, golden hour lighting, displaying natural patterns, textures, and authentic wildlife behavior",
        "majestic {subject} captured in pristine natural setting, dramatic lighting highlighting distinctive features, colors, and characteristic poses"
    ],
    'architecture': [
        "architectural details of {subject}, emphasizing structural elements, historical significance, cultural heritage, and traditional craftsmanship",
        "stunning view of {subject} showcasing design elements, cultural importance, architectural style, and historical context",
        "detailed perspective of {subject} highlighting construction techniques, artistic elements, and cultural significance"
    ],
    'landscape': [
        "breathtaking landscape of {subject} with dramatic lighting, natural beauty, scenic vistas, and environmental characteristics",
        "panoramic view of {subject} featuring natural elements, atmospheric conditions, geological features, and scenic beauty",
        "serene landscape of {subject} capturing natural lighting, topographical features, and environmental essence"
    ],
    'cultural': [
        "traditional {subject} showcasing cultural heritage, craftsmanship, artistic techniques, and historical significance",
        "authentic representation of {subject} highlighting cultural practices, traditional methods, and artistic expression",
        "detailed view of {subject} emphasizing cultural importance, traditional craftsmanship, and historical context"
    ],
    'food': [
        "traditional preparation of {subject} showing culinary techniques, ingredients, presentation, and cultural food heritage",
        "authentic {subject} highlighting cooking methods, traditional ingredients, presentation style, and cultural significance",
        "detailed view of {subject} showcasing preparation process, traditional techniques, and cultural culinary practices"
    ]
}

# Function to categorize folder and generate detailed caption
def categorize_and_caption(folder_name):
    """Categorize the folder and return appropriate category and subject"""
    folder_lower = folder_name.lower()
    
    # Wildlife categories
    if any(word in folder_lower for word in ['junglefowl', 'elephant', 'cat', 'fishing_cat', 'bird']):
        return 'wildlife', folder_name.replace('_', ' ')
    
    # Architecture categories  
    elif any(word in folder_lower for word in ['fort', 'temple', 'mosque', 'station', 'tower', 'airport', 'hospital', 'houses']):
        return 'architecture', folder_name.replace('_', ' ')
    
    # Landscape categories
    elif any(word in folder_lower for word in ['falls', 'beach', 'bay', 'plains', 'mountain', 'lake', 'island', 'rainforest', 'national_park']):
        return 'landscape', folder_name.replace('_', ' ')
    
    # Cultural/craft categories
    elif any(word in folder_lower for word in ['weaving', 'craft', 'carving', 'lace', 'dance', 'drummers', 'perahera']):
        return 'cultural', folder_name.replace('_', ' ')
    
    # Food categories
    elif any(word in folder_lower for word in ['thiyal', 'kottu', 'roti', 'kiribath', 'lamprais', 'hopper']):
        return 'food', folder_name.replace('_', ' ')
    
    # Default to cultural
    else:
        return 'cultural', folder_name.replace('_', ' ')

# Function to generate enhanced caption for an image
def generate_detailed_caption(image_path, folder_name):
    try:
        # Get category and subject
        category, subject = categorize_and_caption(folder_name)
        
        if processor is None or model is None:
            # Fallback: create detailed description based on folder name and category
            template = random.choice(caption_templates[category])
            return f"{folder_name.replace('_', ' ')}: {template.format(subject=subject)}"
        
        image = Image.open(image_path).convert("RGB")
        
        # Generate base caption with BLIP
        inputs = processor(image, return_tensors="pt")
        
        with torch.no_grad():
            # Generate with more detailed parameters
            out = model.generate(**inputs, max_length=100, num_beams=4, temperature=0.7)
            base_caption = processor.decode(out[0], skip_special_tokens=True)
        
        # Enhance the caption with template
        template = random.choice(caption_templates[category])
        enhanced_description = template.format(subject=subject)
        
        # Combine base caption with enhanced template
        if len(base_caption) > 10:  # If we got a good caption
            final_caption = f"{folder_name.replace('_', ' ')}: {base_caption}, {enhanced_description}"
        else:
            final_caption = f"{folder_name.replace('_', ' ')}: {enhanced_description}"
        
        return final_caption
        
    except Exception as e:
        # Fallback description with template
        category, subject = categorize_and_caption(folder_name)
        template = random.choice(caption_templates[category])
        return f"{folder_name.replace('_', ' ')}: {template.format(subject=subject)}"

# Loop through folders and images
print(f"Scanning directory: {root_dir}")
for folder_name in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder_name)
    if not os.path.isdir(folder_path):
        print(f"Skipping non-directory: {folder_name}")
        continue

    print(f"Processing folder: {folder_name}")
    try:
        images = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        print(f"Found {len(images)} images in {folder_name}")
    except PermissionError:
        print(f"Permission denied accessing {folder_name}")
        continue

    for img_name in images:
        img_path = os.path.join(folder_path, img_name)
        txt_path = os.path.splitext(img_path)[0] + ".txt"

        caption = generate_detailed_caption(img_path, folder_name)

        # Create/overwrite caption file
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(caption)

        print(f"✓ {img_name}")  # Simplified output

print("✅ Image captioning complete.")
