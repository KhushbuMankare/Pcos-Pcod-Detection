🔬 HerHelth AI – PCOS Detection

Polycystic Ovary Syndrome (PCOS) is a common hormonal disorder affecting women of reproductive age, often causing irregular periods, infertility, and metabolic issues. HerHelth AI is a machine learning–based solution designed to assist in the early detection of PCOS using medical and lifestyle data.

🚀 Features

✅ Early prediction of PCOS likelihood using ML models

✅ User-friendly interface built with Streamlit

✅ Real-time results with high accuracy

✅ Scalable for larger datasets and extended features

🛠️ Tech Stack

Language: Python

Libraries: NumPy, Pandas, scikit-learn, Streamlit, Pickle

Tools: Jupyter Notebook, Git

📂 Project Structure
├── modelfinal1.pkl        # Trained ML model  
├── app.py                 # Streamlit web application  
├── requirements.txt       # Dependencies  
├── dataset.csv            # Training dataset (if available)  
└── README.md              # Project documentation  

⚙️ Installation & Usage

Clone the repository:

git clone https://github.com/your-username/pcos-detection.git
cd pcos-detection


Install dependencies:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run app.py


Open the app in your browser at:
👉 http://localhost:8501

📊 How It Works

Input: User enters medical details (BMI, hormone levels, cycle history, etc.)

Processing: Data is preprocessed and passed into the ML model

Output: Prediction of PCOS likelihood (Yes/No + probability score)

🎯 Outcome

HerHelth AI helps healthcare professionals and individuals gain early insights into PCOS risk, supporting proactive health management and timely medical intervention.

🤝 Contributing

Contributions are welcome! Fork the repo, make your changes, and submit a pull request.

📜 License

This project is licensed under the MIT License.
