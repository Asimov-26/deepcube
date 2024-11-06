import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file containing the metrics
data = pd.read_csv("metrics.csv")

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(data['Epoch'], data['Target Cost'], label='Target Cost')
plt.plot(data['Epoch'], data['Squared Loss'], label='Squared Loss')
plt.plot(data['Epoch'], data['Cross Entropy Loss'], label='Cross Entropy Loss')

# Add labels and title
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Losses Over Epochs')
plt.legend()

# Show the plot
plt.show()
