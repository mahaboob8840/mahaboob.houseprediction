# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import joblib
# import pandas as pd
# import numpy as np
# import os

# app = Flask(__name__)
# CORS(app)

# MODEL_PATH = 'house_price_prediction_model.pkl'

# if os.path.exists(MODEL_PATH):
#     model = joblib.load(MODEL_PATH)
#     print(f"✅ Model loaded: {MODEL_PATH}")
# else:
#     print(f"❌ Model not found")
#     model = None

# # --- NEW: Add this Home Route to fix the 404 Browser check ---
# # --- REPLACE THE predict() FUNCTION IN backend.py WITH THIS ---

# @app.route('/predict', methods=['POST'])
# def predict():
#     if not model:
#         return jsonify({'error': 'Model not loaded'}), 500

#     try:
#         data = request.get_json()
        
#         # 1. Get the real value from the frontend
#         if 'sqft_living' not in data:
#             return jsonify({'error': 'Missing sqft_living'}), 400
        
#         sqft_value = float(data['sqft_living'])

#         # 2. Create a DataFrame with ALL the columns your model expects
#         # We use default values (averages) for the missing data.
#         # YOU MUST ADD EVERY COLUMN NAME LISTED IN YOUR ERROR MESSAGE HERE.
#         input_data = pd.DataFrame({
#             'sqft_living': [sqft_value],  # Real value from user
#             'bedrooms': [3],              # Default: 3 bedrooms
#             'bathrooms': [2.0],           # Default: 2 bathrooms
#             'floors': [1],                # Default: 1 floor
#             'waterfront': [0],            # Default: No waterfront
#             'view': [0],                  # Default: No view
#             'condition': [3],             # Default: Average condition
#             'sqft_above': [sqft_value],   # Default: Same as living
#             'sqft_basement': [0],         # Default: No basement
#             'yr_built': [1990],           # Default: Built in 1990
#             'yr_renovated': [0],          # Default: Never renovated
#             # ... ADD ANY OTHER MISSING COLUMNS HERE ...
#         })
        
#         # 3. Ensure columns are in the exact order the model expects
#         # (This handles the "Feature names seen at fit time" error)
#         # Note: If this still fails, we might need to match the column order strictly.
        
#         prediction = model.predict(input_data)
        
#         return jsonify({
#             'status': 'success',
#             'predicted_price': float(prediction[0])
#         })

#     except Exception as e:
#         print(f"Prediction Error: {e}")
#         return jsonify({'error': str(e)}), 500

# # --- 4. Run the Server ---
# if __name__ == '__main__':
#     # host='0.0.0.0' makes it accessible on your local network
#     print("🚀 Server is running on http://127.0.0.1:8000") 
#     app.run(debug=True, port=5000, host='0.0.0.0')








from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Try the simple model first, fall back to creating one
MODEL_PATH_SIMPLE = 'house_price_model_simple.pkl'
MODEL_PATH_OLD = 'house_price_prediction_model.pkl'

def load_or_create_model():
    """Load existing model or create a simple one"""
    # First try the simple model
    if os.path.exists(MODEL_PATH_SIMPLE):
        model = joblib.load(MODEL_PATH_SIMPLE)
        print(f"✅ Simple model loaded: {MODEL_PATH_SIMPLE}")
        features = ['sqft_living', 'bedrooms', 'bathrooms', 'condition']
        return model, features, "simple"
    
    # Try the old model
    elif os.path.exists(MODEL_PATH_OLD):
        model = joblib.load(MODEL_PATH_OLD)
        print(f"⚠️ Old model loaded: {MODEL_PATH_OLD}")
        # Try to determine features from model
        try:
            # This works for scikit-learn models
            if hasattr(model, 'feature_names_in_'):
                features = model.feature_names_in_.tolist()
            else:
                # Default to our simple features
                features = ['sqft_living', 'bedrooms', 'bathrooms', 'condition']
        except:
            features = ['sqft_living', 'bedrooms', 'bathrooms', 'condition']
        return model, features, "old"
    
    else:
        # Create a simple model
        print(f"❌ No model found. Creating a simple model...")
        from sklearn.linear_model import LinearRegression
        
        # Create and train a simple model
        model = LinearRegression()
        X_dummy = np.array([
            [1000, 2, 1.0, 3],
            [1500, 3, 2.0, 3],
            [2000, 3, 2.0, 4],
            [2500, 4, 2.5, 4],
            [3000, 4, 3.0, 5]
        ])
        y_dummy = np.array([300000, 450000, 600000, 750000, 900000])
        model.fit(X_dummy, y_dummy)
        
        # Save it
        joblib.dump(model, MODEL_PATH_SIMPLE)
        print(f"✅ Simple model created and saved to {MODEL_PATH_SIMPLE}")
        
        features = ['sqft_living', 'bedrooms', 'bathrooms', 'condition']
        return model, features, "created"

# Load model
model, model_features, model_type = load_or_create_model()
print(f"📋 Model features expected: {model_features}")
print(f"🔧 Model type: {model_type}")

@app.route('/')
def home():
    return jsonify({
        'message': '🏠 House Price Prediction API',
        'status': 'active',
        'model_type': model_type,
        'features_required': model_features,
        'endpoints': {
            'predict': 'POST /predict',
            'health': 'GET /',
            'features': 'GET /features'
        }
    })

@app.route('/features', methods=['GET'])
def get_features():
    """Return the features required by the model"""
    return jsonify({
        'features_required': model_features,
        'model_type': model_type,
        'example_request': {
            'sqft_living': 2000,
            'bedrooms': 3,
            'bathrooms': 2.0,
            'condition': 3
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print(f"📥 Received request data: {data}")
        
        # Set default values for missing features
        defaults = {
            'sqft_living': 1500,
            'bedrooms': 3,
            'bathrooms': 2.0,
            'condition': 3,
            'floors': 1,
            'waterfront': 0,
            'view': 0,
            'sqft_above': None,
            'sqft_basement': 0,
            'yr_built': 1990,
            'yr_renovated': 0
        }
        
        # Prepare input dictionary
        input_dict = {}
        
        # For simple model (4 features)
        if len(model_features) == 4 and all(f in ['sqft_living', 'bedrooms', 'bathrooms', 'condition'] for f in model_features):
            # Get values or use defaults
            for feature in model_features:
                if feature in data:
                    input_dict[feature] = [float(data[feature])]
                else:
                    input_dict[feature] = [float(defaults[feature])]
                    print(f"⚠️ Using default for {feature}: {defaults[feature]}")
        else:
            # For complex models, try to get all features
            for feature in model_features:
                if feature in data:
                    input_dict[feature] = [float(data[feature])]
                elif feature in defaults:
                    input_dict[feature] = [float(defaults[feature])]
                    print(f"⚠️ Using default for {feature}: {defaults[feature]}")
                else:
                    # If feature not in defaults, use a reasonable default
                    if 'sqft' in feature:
                        input_dict[feature] = [float(data.get('sqft_living', 1500))]
                    elif 'bed' in feature:
                        input_dict[feature] = [3.0]
                    elif 'bath' in feature:
                        input_dict[feature] = [2.0]
                    else:
                        input_dict[feature] = [0.0]
        
        # Create DataFrame with exact feature names
        input_df = pd.DataFrame(input_dict)
        
        # Ensure columns are in correct order
        input_df = input_df[model_features]
        
        print(f"📋 Input DataFrame shape: {input_df.shape}")
        print(f"📋 Input DataFrame columns: {input_df.columns.tolist()}")
        print(f"📋 Input values: {input_df.values.tolist()}")
        
        # Make prediction
        prediction = model.predict(input_df)
        predicted_price = float(prediction[0])
        
        # Format the response
        response = {
            'status': 'success',
            'predicted_price': round(predicted_price, 2),
            'input_features': input_df.iloc[0].to_dict(),
            'model_type': model_type,
            'features_used': model_features,
            'formatted_price': f"${predicted_price:,.2f}"
        }
        
        print(f"✅ Prediction: ${predicted_price:,.2f}")
        return jsonify(response)
        
    except Exception as e:
        error_msg = f"Prediction Error: {str(e)}"
        print(f"❌ {error_msg}")
        return jsonify({
            'status': 'error',
            'error': error_msg,
            'features_required': model_features,
            'tip': 'Make sure to provide all required features or run hacketon_houseprediction_simple.py first'
        }), 400

@app.route('/predict_simple', methods=['POST'])
def predict_simple():
    """Simplified endpoint that only requires sqft_living"""
    try:
        data = request.get_json()
        
        if 'sqft_living' not in data:
            return jsonify({'error': 'sqft_living is required'}), 400
        
        sqft = float(data['sqft_living'])
        bedrooms = float(data.get('bedrooms', 3))
        bathrooms = float(data.get('bathrooms', 2.0))
        condition = float(data.get('condition', 3))
        
        # Create input for simple model
        input_data = pd.DataFrame({
            'sqft_living': [sqft],
            'bedrooms': [bedrooms],
            'bathrooms': [bathrooms],
            'condition': [condition]
        })
        
        prediction = model.predict(input_data)
        predicted_price = float(prediction[0])
        
        return jsonify({
            'status': 'success',
            'predicted_price': round(predicted_price, 2),
            'formatted_price': f"${predicted_price:,.2f}",
            'input': {
                'sqft_living': sqft,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'condition': condition
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n{'='*60}")
    print(f"🏠 House Price Prediction API")
    print(f"{'='*60}")
    print(f"✅ Model loaded: {model_type} model with {len(model_features)} features")
    print(f"📋 Features: {model_features}")
    print(f"🌐 Server starting on: http://127.0.0.1:{port}")
    print(f"🌐 Also available on: http://localhost:{port}")
    print(f"{'='*60}")
    print(f"📝 Test with: curl -X POST http://127.0.0.1:{port}/predict_simple \\")
    print(f"  -H 'Content-Type: application/json' \\")
    print(f"  -d '{{\"sqft_living\": 2000}}'")
    print(f"{'='*60}")
    
    # Disable the auto reloader to avoid Windows socket-fromfd errors
    app.run(debug=True, port=port, host='0.0.0.0', use_reloader=False)






















    # End of file: removed duplicate Flask stub to avoid conflicts on restart