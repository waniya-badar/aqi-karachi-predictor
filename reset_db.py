from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()
client = MongoClient(os.getenv("MONGODB_URI"))
db = client[os.getenv("MONGODB_DB_NAME")]

# Delete all documents in 'features' collection
db.features.delete_many({})  

client.close()
print("All records deleted")
