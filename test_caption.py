import os
import sys
sys.path.append('.')

# Test the caption generation function
def test_caption():
    folder_name = 'ceylon_junglefowl'
    
    # Test with fallback (no model)
    from vision import categorize_and_caption, caption_templates
    import random
    
    category, subject = categorize_and_caption(folder_name)
    template = random.choice(caption_templates[category])
    caption = f"{folder_name.replace('_', ' ')}: {template.format(subject=subject)}"
    
    print("Test caption generated:")
    print(caption)
    print()
    print(f"Category: {category}")
    print(f"Subject: {subject}")

if __name__ == "__main__":
    test_caption()
