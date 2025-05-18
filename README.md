# 🏡 Real Estate Analyzer

**Real Estate Analyzer** is a Django-based web application designed to analyze and compare real estate data across different areas. This intelligent platform enables users to interact with the system using conversational queries powered by NLP, providing insightful visualizations and comparisons for informed decision-making.

---

## 🚀 Features

- **📊 Area Analysis**  
  Analyze real estate trends, metrics, and insights for a selected geographical area.

- **⚖️ Area Comparison**  
  Compare multiple real estate metrics between two different areas for better investment decisions.

- **🧠 Natural Language Processing**  
  Integrates with **Gemini** to interpret and process user queries through conversational language.

- **📈 Data Visualization**  
  Presents analytical results and insights in a user-friendly and visual format.

---

## 🛠️ Tech Stack

### 🔧 Backend
- **Django**
- **Django REST Framework (DRF)**

### 💻 Frontend
- **React.js** (assumed standard SPA setup)

### 🤖 NLP Integration
- **Gemini API**

### 📊 Data Handling
- **Pandas**
- **Excel Files**

### 🧪 API Testing
- **Postman** or equivalent API testing tools

---

## 📁 Project Structure (Sample)

real_estate_analyzer/
│
├── backend/
│ ├── analyzer/
│ ├── api/
│ ├── manage.py
│ └── ...
│
├── frontend/
│ └── react-app/
│ ├── src/
│ └── ...
│
├── data/
│ └── real_estate_data.xlsx
│
└── README.md


## 📌 How to Run

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/your-username/real-estate-analyzer.git
   cd real-estate-analyzer
Setup Backend

bash
Copy
Edit
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python manage.py runserver
Setup Frontend (React)

bash
Copy
Edit
cd frontend/react-app
npm install
npm start
Access Application
Visit: http://localhost:3000

💬 API Testing
Use Postman to test all available endpoints.

Example endpoints:

GET /api/area-analysis/?location=wakad

POST /api/compare-areas/

POST /api/query/ with natural language query


🧑‍💻 Contributing
Contributions are welcome! Please fork the repo and submit a pull request.

