import requests
import pandas as pd

# Step 1: Define the API URL
URL = "https://hydrologic-valvar-hang.ngrok-free.dev/social-media/user/getAll"

# Step 2: Fetch the JSON response from the API
response = requests.get(URL)
response.raise_for_status()  # raises an error if request fails

data = response.json()

# Step 3: Extract the list under the key 'userResponses'
user_data = data.get("userResponses", [])

# Step 4: Convert JSON list to a DataFrame
df = pd.DataFrame(user_data)

# Step 5: Save DataFrame to CSV
df.to_csv("user_data2.csv", index=False, encoding="utf-8")

print(f"Data fetched successfully! {len(df)} records saved to 'user_data.csv'")