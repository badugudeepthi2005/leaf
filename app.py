from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json


app = Flask(__name__)
CORS(app)


# Load your Keras model
model = load_model('model.h5')


# Load class mappings saved during training
with open('class_mapping.json', 'r') as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}


# Remedies dictionary
REMEDIES = {
    "Bacterial spot": {
        "en": "Use copper-based bactericides. Remove infected leaves and avoid overhead watering.",
        "te": "తామోటా మొక్కల్లో సోడియం బేస్డ్ బెక్టీరియాసిడ్ వాడండి. అంటుకునిన ఆకులను తీసివేయండి మరియు పై నుండి నీరు పోసకండి."
    },
    "Early blight": {
        "en": "Apply Mancozeb or Chlorothalonil fungicides. Remove and destroy infected debris.",
        "te": "మాంకోజెబ్ లేదా క్లోరోథాలొనిల్ ఫంగిసైడ్లు వాడండి. అంటుకున్న భాగాలను తీసివేసి ధ్వంసం చేయండి."
    },
    "Late blight": {
        "en": "Use Metalaxyl and avoid overhead irrigation. Remove diseased plants promptly.",
        "te": "మెటలాక్సిల్ వాడండి మరియు పై నుంచి నీరు పోసొద్దు. అంటుకున్న మొక్కలను తక్షణమే తీసివేయండి."
    },
    "Leaf Mold": {
        "en": "Maintain good greenhouse ventilation. Apply fungicides like Amistar or Serenade.",
        "te": "గుడ్ గ్రీన్హౌస్ వాయు ప్రసరణ ఉంచండి. అమిస్తార్ లేదా సిరెనేడ్ వంటి ఫంగిసైడ్లు వాడండి."
    },
    "Septoria leaf spot": {
        "en": "Remove infected leaves regularly. Use fungicides containing chlorothalonil.",
        "te": "అంటున్న ఆకులను తరచుగా తీసివేయండి. క్లోరోథాలొనిల్ కలిగిన ఫంగిసైడ్లు వాడండి."
    },
    "Spider mites Two spotted spider mite": {
        "en": "Use insecticidal soaps or neem oil. Encourage predator mites.",
        "te": "ఇన్‌సెక్టిసైడల్ సాబున్లు లేదా నిమ్మ ఆయిల్ వాడండి. ప్రీడేటర్ మైట్స్ ను ప్రోత్సహించండి."
    },
    "Leaf curl virus": {
        "en": "Control whiteflies with insecticides like Pyron. Remove infected plants immediately.",
        "te": "వైట్‌డిల్స్ ని పైరాన్ వంటి కీటనాశకాలతో నియంత్రించండి. అంటుకున్న మొక్కలను వెంటనే తొలగించండి."
    },
    "Target Spot": {
        "en": "Use protectant fungicides like Azoxystrobin, Mancozeb, or Chlorothalonil. Remove infected plant debris, practice crop rotation, and ensure proper air circulation to reduce humidity.",
        "te": "అశోక్సిస్ట్రోబిన్, మాంకోజెబ్ లేదా క్లోరోథాలోనిల్ వంటి రక్షణ ఫంగిసైడ్‌లను వాడండి. అంటుకున్న మొక్క పదార్థాలను తీసివేయండి, పంట మలుప λάారని పాటించండి మరియు తగిన గాలి పరిభ్రమణకు అవకాశం ఇస్తూ ఆర్గానిక వాతావరణాన్ని మెరుగుపరచండి."
    },
    "Anthracnose": {
        "en": "Use disease-free seed and rotate crops. Apply copper fungicides preventively.",
        "te": "వ్యాధి రహిత విత్తనాలు వాడండి మరియు పంటల మార్పిడి చేయండి. కోపర్ ఫంగిసైడ్లను ముందుగానే వాడండి."
    },
    "Tomato YellowLeaf Curl Virus": {
        "en": "Control whiteflies using insecticides like Imidacloprid or Thiamethoxam. Remove infected plants immediately. Use yellow sticky traps and grow barrier crops like sorghum or maize.",
        "te": "ఇమిడాక్లోప్రిడ్ లేదా థియామెథాక్సామ్ వంటి కీటనాశకాలతో వైట్‌లైఫ్లను నియంత్రించండి. అంటుకున్న మొక్కలను వెంటనే తీసివేయండి. పసుపు స్టికీ ట్రాప్స్ వాడండి మరియు సొరుగం, మక్కజొన్న వంటి అడ్డంకి పంటలను పెంచండి."
}
    # Add more classes with similar treatment strategies as needed
}

# Mapping for predicted classes to remedy dictionary keys
CLASS_KEY_MAPPING = {
    'Tomato_Bacterial_spot': 'Bacterial spot',
    'bell Bacterial spot': 'Bacterial spot',
    'Tomato_Early_blight': 'Early blight',
    'Potato___Early_blight': 'Early blight',
    'Tomato_Late_blight': 'Late blight',
    'Potato___Late_blight': 'Late blight',
    'Tomato_Leaf_Mold': 'Leaf Mold',
    'Tomato_Septoria_leaf_spot': 'Septoria leaf spot',
    'target_spot': 'Target Spot',
    'Tomato_Target_Spot': 'Target Spot',
    'Tomato_Yellow_Leaf_Curl_Virus': 'Tomato YellowLeaf Curl Virus',
    'PlantVillage': 'Spider mites Two spotted spider mite',
    # Add more if necessary based on your dataset
}

@app.route("/", methods=["GET"])
def home():
    return render_template("index5.html")

# (All your other routes, like /predict)
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    lang = request.form.get('lang', 'en')
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file.save('temp.jpg')  # Save temporary file

    # Preprocess for prediction
    img = image.load_img('temp.jpg', target_size=(224, 224))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    pred_idx = np.argmax(preds, axis=1)[0]
    pred_class = idx_to_class[pred_idx].strip()

    # Map predicted class to remedy key if necessary
    pred_class = CLASS_KEY_MAPPING.get(pred_class, pred_class)

    print("Predicted class:", repr(pred_class))
    print("Remedy keys:", list(REMEDIES.keys()))

    remedy = REMEDIES.get(pred_class, {'en': 'No remedy found.', 'te': 'చికిత్స లేదు.'})

    return jsonify({'name': pred_class, 'remedy': remedy[lang]})


if __name__ == "__main__":
     app.run(debug=True, host="0.0.0.0", port=5000)
