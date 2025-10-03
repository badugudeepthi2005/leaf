import json

# Load your dataset manifest JSON
with open('images_manifest.json') as f:
    data = json.load(f)

# Extract unique class names
unique_classes = sorted(list(set(item['class'] for item in data)))

# Create mapping: class_name -> index
class_mapping = {class_name: idx for idx, class_name in enumerate(unique_classes)}

# Save mapping as JSON file
with open('class_mapping.json', 'w') as f:
    json.dump(class_mapping, f, indent=2)

print("class_mapping.json created with", len(class_mapping), "classes.")
