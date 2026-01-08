# Crop Price Prediction and Farm Management System

A comprehensive Flask-based web application for Karnataka farmers to predict crop prices, get crop recommendations, analyze market trends, and access farming tools.
![Frontend](image.png)


## Features

### 🏷️ Price Prediction
- Machine learning-powered crop price forecasting
- Historical data analysis for accurate predictions
- District-wise market price insights
- Real-time price updates with web scraping

### 🌾 Crop Rotation & Recommendations
- AI-powered crop suggestion based on soil type, season, and previous crops
- Profit maximization recommendations
- Smart rotation to prevent pest cycles
- District-specific crop suitability analysis

### 📊 Market Trends & High Demand Crops
- Monthly high-demand crop analysis
- Market trend predictions
- Export potential identification
- Seasonal price pattern analysis

### 🌤️ Weather Advisory
- Real-time weather data integration (OpenWeatherMap API)
- Farming advisory based on weather conditions
- District-wise weather forecasts
- Crop-specific weather recommendations

### 💰 Profit Calculator
- Investment vs. revenue analysis
- Yield-based profit calculations
- Cost optimization suggestions
- Break-even analysis

### 🧪 Fertilizer Calculator
- NPK requirement calculations
- Cost-effective fertilizer recommendations
- Crop-specific nutrient planning

### 🔬 Disease Detection
- Symptom-based crop disease identification
- Treatment recommendations
- Prevention strategies

## Technology Stack

- **Backend:** Flask (Python)
- **Frontend:** HTML, CSS, JavaScript, Bootstrap
- **Machine Learning:** XGBoost, Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Web Scraping:** BeautifulSoup, Selenium
- **Database:** CSV-based data storage
- **APIs:** OpenWeatherMap for weather data


## Project Structure

```
├── app.py                      # Main Flask application
├── train_final_model.py        # Model training script
├── multi_crop_scraper.py       # Web scraper for price data
├── final_complete_data.csv     # Historical crop price data
├── final_crop_model2.pkl       # Trained ML model
├── final_encoders.pkl          # Label encoders
├── templates/                  # HTML templates
│   ├── index.html
│   ├── welcome.html
│   ├── crop_rotation.html
│   ├── market_trends.html
│   └── ...
├── static/                     # CSS, JS, images
└── requirements.txt            # Python dependencies
```
