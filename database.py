from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")

# Create / Connect Database
db = client["student_performance_db"]

# Collections
students = db["Students"]
predictions = db["Predictions"]

print("MongoDB Connected Successfully!")