# Stock Market Prediction System

## Project Overview  
A web-based application that uses deep learning (LSTM) to predict stock market trends. Built with Flask, it integrates real-time stock data fetching, technical indicators, and interactive visualizations.

## Features  
- Predict stock price trends using an LSTM model  
- Fetch stock data using yfinance  
- Display interactive charts with Plotly  
- User-friendly web interface with Flask  
- Historical prediction records  

## Installation & Setup

### Prerequisites  
- Python 3.12 or later  
- Git  
- Virtual environment tool (optional but recommended)

### Steps

1. **Clone the repository**  
   bash  
   git clone https://github.com/yourusername/your-repo-name.git  
   cd your-repo-name  


2. **Create and activate a virtual environment**

   bash
   python -m venv venv  
   # Windows  
   .\venv\Scripts\activate  
   # macOS/Linux  
   source venv/bin/activate  
   

3. **Install dependencies**

   bash
   pip install -r requirements.txt  
   

4. **Create required folders**
   Before running the app, create the following directories inside your project folder:

   bash
   mkdir templates
   mkdir static
   

5. **Add your HTML templates and static files**

   * Put all your HTML files (e.g., `index.html`, `login.html`, `predict.html`) inside the `templates/` folder.
   * Place CSS, images (like `nn.webp`), and JavaScript files inside the `static/` folder.

6. **Run the Flask app**

   bash
   python app.py  
   

7. **Open your browser and go to**

   
   http://127.0.0.1:5000
   

## Project Structure

* `app.py`: Main Flask application
* `model.pkl`, `scaler.pkl`: Trained model and scaler files
* `templates/`: HTML templates
* `static/`: CSS, images, and JavaScript files
* `requirements.txt`: Python dependencies

## Usage

* Enter a stock ticker symbol and date range
* View historical data and predicted trends
* Analyze technical indicators like Moving Averages (MA) and Relative Strength Index (RSI)

## Contribution

Feel free to open issues or submit pull requests.

## License

Specify your license here (e.g., MIT)

