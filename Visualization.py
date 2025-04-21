import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'ev_charging_patterns.csv'
df = pd.read_csv(file_path)

# Set the style for the plots
sns.set(style="whitegrid")

# 1. Distribution of Charging Duration
plt.figure(figsize=(10, 6))
sns.histplot(df['Charging Duration (hours)'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Charging Duration (Hours)', fontsize=16)
plt.xlabel('Charging Duration (hours)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.show()

# 2. Charging Rate vs Energy Consumed
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Charging Rate (kW)', y='Energy Consumed (kWh)', data=df, color='orange')
plt.title('Charging Rate vs Energy Consumed', fontsize=16)
plt.xlabel('Charging Rate (kW)', fontsize=14)
plt.ylabel('Energy Consumed (kWh)', fontsize=14)
plt.show()

# 3. Charging Costs by Charger Type
plt.figure(figsize=(10, 6))
sns.boxplot(x='Charger Type', y='Charging Cost (USD)', data=df, palette="Set2")
plt.title('Charging Costs by Charger Type', fontsize=16)
plt.xlabel('Charger Type', fontsize=14)
plt.ylabel('Charging Cost (USD)', fontsize=14)
plt.show()

# 4. Energy Consumed vs Temperature
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Temperature (°C)', y='Energy Consumed (kWh)', data=df, color='green')
plt.title('Energy Consumed vs Temperature (°C)', fontsize=16)
plt.xlabel('Temperature (°C)', fontsize=14)
plt.ylabel('Energy Consumed (kWh)', fontsize=14)
plt.show()

# 5. Charging Duration by Day of Week
plt.figure(figsize=(10, 6))
sns.boxplot(x='Day of Week', y='Charging Duration (hours)', data=df, palette="coolwarm")
plt.title('Charging Duration by Day of Week', fontsize=16)
plt.xlabel('Day of Week', fontsize=14)
plt.ylabel('Charging Duration (hours)', fontsize=14)
plt.show()

# 6. Charging Duration by User Type
plt.figure(figsize=(10, 6))
sns.boxplot(x='User Type', y='Charging Duration (hours)', data=df, palette="Set3")
plt.title('Charging Duration by User Type', fontsize=16)
plt.xlabel('User Type', fontsize=14)
plt.ylabel('Charging Duration (hours)', fontsize=14)
plt.show()

# 7. Energy Consumed by User Type
plt.figure(figsize=(10, 6))
sns.boxplot(x='User Type', y='Energy Consumed (kWh)', data=df, palette="Set3")
plt.title('Energy Consumed by User Type', fontsize=16)
plt.xlabel('User Type', fontsize=14)
plt.ylabel('Energy Consumed (kWh)', fontsize=14)
plt.show()

# 8. Charging Cost by User Type
plt.figure(figsize=(10, 6))
sns.boxplot(x='User Type', y='Charging Cost (USD)', data=df, palette="Set3")
plt.title('Charging Cost by User Type', fontsize=16)
plt.xlabel('User Type', fontsize=14)
plt.ylabel('Charging Cost (USD)', fontsize=14)
plt.show()

# 9. Count of Each User Type
plt.figure(figsize=(10, 6))
sns.countplot(x='User Type', data=df, palette="Set2")
plt.title('Count of Each User Type', fontsize=16)
plt.xlabel('User Type', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.show()
