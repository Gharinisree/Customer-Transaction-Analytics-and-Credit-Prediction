# Customer Transaction Analytics & Credit Risk Prediction

A full-stack Flask web application integrated with an SQLite database to track customer billing histories, analyze payment behavior patterns, and deliver real-time machine learning credit risk evaluations.

## Key Features
* **Analytics Dashboard:** Real-time visual monitoring of customer spending, billing cycles, and transaction statuses.
* **ML Credit Risk Engine:** Machine learning model (`credit_model.pkl`) to evaluate customer risk profiles instantly.
* **Database Management:** Lightweight, local relational data tracking powered by SQLite.

## Project Structure
* `app.py` - Flask web application backend and routing.
* `train_model.py` - ML model training pipeline and artifact generation.
* `seed_local_db.py` - SQLite database initialization and seeding script.
* `credit_model.pkl` - Trained machine learning model binary.
* `requirements.txt` - Project dependency specifications.

## Local Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Gharinisree/Customer-Transaction-Analytics-and-Credit-Prediction.git](https://github.com/Gharinisree/Customer-Transaction-Analytics-and-Credit-Prediction.git)
   cd Customer-Transaction-Analytics-and-Credit-Prediction
   
