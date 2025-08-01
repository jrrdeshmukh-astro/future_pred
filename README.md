## Modern Mercantilism: Thesis Statements

### 1. A Non-Regional War Could Emerge in the 2040s
Predictability initially improves but then declines sharply around 2040, signaling a period of increased volatility and risk. This dip may indicate rising geopolitical tensions or global instability, potentially culminating in a non-regional conflict before by 2045.

### 2. Global Mobility Has Peaked and Is Now Declining
After a phase of rapid growth, global mobility metrics show signs of contraction. This trend suggests growing barriers to international movement and trade, with implications for global economic integration.

### 3. Human Cooperation Remains Stable Despite Declining Mobility
While global mobility is decreasing, cooperation metrics remain steady. This resilience indicates that nations are adapting and maintaining collaborative efforts even in the face of reduced physical connectivity.

    
![png](mcp.png)
    

## Implementation Notes

Signal processing and forecasting techniques are applied to analyze the Z-scores of Mobility, Cooperation, and Predictability. The results are visualized to identify trends and potential future scenarios.

### What is mobility, cooperation, and predictability?
Mobility refers to the ease of movement across borders, including trade and migration. 
Cooperation encompasses collaborative efforts between nations, such as treaties and partnerships. 
Predictability relates to the stability and reliability of international relations and economic conditions.

### Why use these paradigms?
These paradigms are essential for understanding global dynamics. Mobility metrics help assess economic interdependence, cooperation signals indicate the strength of international relationships, and predictability metrics provide insights into geopolitical stability. Together, they offer a comprehensive view of the current and future state of global interactions.


### The Signals used in this analysis are:
- Cooperation Metrics
    - DTP3 Vaccination Coverage (%)
    - Annual CO2 Emissions (Gigatons)
    - Global Literacy Rate (%)
    - Global Life Expectancy (Years)
- Mobility Metrics
    - Global Trade Volume (% of GDP)
    - Internet Penetration Rate (% Global Pop)
    - Cross-border Internet Traffic (Exabytes)
    - International Migrant Stock (Millions)
-  Predictability Metrics
    - Geopolitical Risk Index (GPR)
    - SPY Volatility (%)

Methodology:
1. **Data Preparation**: Load the dataset and calculate Z-scores for each signal.
2. **GARCH Modeling**: Fit GARCH models to the Z-scores of Mobility, Cooperation, and Predictability to capture volatility and trends.
3. **LSTM Forecasting**: Use LSTM networks to predict future values of the Z-scores based on historical data.
4. **Visualization**: Plot the Z-scores, GARCH forecasts, and LSTM predictions to visualize trends and potential future scenarios.
5. **Analysis**: Interpret the results to draw conclusions about the future of global mobility, cooperation, and predictability.


```python
!pip install pandas numpy scipy arch scikit-learn tensorflow matplotlib seaborn ipython
import pandas as pd
import numpy as np
from scipy.stats import zscore # Import zscore function
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image

print("Libraries imported successfully.")

```


    
![png](Coop_LifeExpectancy.png)
    





    
![png](Coop_CO2.png)
    






    
![png](Coop_Literacy.png)
    





    
![png](Coop_Vaccination.png)
    





    
![png](Mob_Cross_Border_Internet.png)
    





    
![png](Mob_Global_Trade.png)
    






    
![png](Mob_IPR.png)
    




    
![png](Mob_Migrant_Stock.png)
    


    
![png](Pred_GPR.png)
    




    
![png](Pred_VIX.png)
    



## 1. Setup and Data Initialization

The Z-scores for Cooperation, Mobility, and Predictability are calculated here, which will then be used for GARCH and LSTM modeling.


```python

# Provided data (from research1.ipynb)
data = {
    'Year': [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2008, 2010, 2015, 2020, 2024],
    # Cooperation Metrics
    'DTP3 Vaccination Coverage (%)': [20, 40, 60, 70, 75, 79, 80, 81, 83, 85, 81, 84],
    'Annual CO2 Emissions (Gigatons)': [15, 18, 19, 22, 24, 26, 29, 31, 32, 34, 34, 37],
    'Global Literacy Rate (%)': [68, 70, 72, 74, 76, 80, 82, 83, 84, 86, 87, 88],
    'Global Life Expectancy (Years)': [58, 60, 62, 64, 66, 67, 69, 70, 70, 72, 71, 73],
    # Mobility Metrics
    'Global Trade Volume (% of GDP)': [27.5, 32.5, 32.5, 37.5, 42.5, 50, 60, 55, 57.5, 57.5, 52.5, 52.5],
    'Internet Penetration Rate (% Global Pop)': [0.05, 0.05, 0.05, 0.5, 5, 6.7, 16, 23, 30, 43, 59.5, 67],
    'Cross-border Internet Traffic (Exabytes)': [0, 0, 0, 0.05, 0.3, 1, 7.5, 17.5, 27.5, 75, 225, 475],
    'International Migrant Stock (Millions)': [85, 95, 105, 150, 165, 175, 190, 210, 220, 245, 280, 305],
    # Predictability Metrics
    'Geopolitical Risk Index (GPR)': [90, 90, 90, 75, 50, 35, 35, 35, 65, 85, 90, 95],
    'SPY Volatility (%)': [9.6, 13.2, 10.3, 9.1, 11.8, 31.4, 8.1, 21.7, 4.6, 19.3, 8.0, 18.9]
}

# Convert data to a pandas DataFrame and set 'Year' as the index
df = pd.DataFrame(data)
df = df.set_index('Year')

print("DataFrame created and 'Year' set as index.")

# Define metric categories
coop_metrics = [
    'DTP3 Vaccination Coverage (%)',
    'Global Literacy Rate (%)',
    'Global Life Expectancy (Years)',
    'Annual CO2 Emissions (Gigatons)'
]

mob_metrics = [
    'Global Trade Volume (% of GDP)',
    'Internet Penetration Rate (% Global Pop)',
    'Cross-border Internet Traffic (Exabytes)',
    'International Migrant Stock (Millions)'
]

red_metrics = [
    'Geopolitical Risk Index (GPR)',
    'SPY Volatility (%)'
]

# Calculate z-scores for each individual metric
for col in df.columns:
    df[f'{col}_ZScore'] = zscore(df[col])

# Adjust 'Predictability' Z-scores so that higher values consistently mean 'better'
df['Geopolitical Risk Index (GPR)_ZScore_Adjusted'] = -df['Geopolitical Risk Index (GPR)_ZScore']
df['SPY Volatility (%)_ZScore_Adjusted'] = -df['SPY Volatility (%)_ZScore']

# List of all Z-score columns, including adjusted ones for predictability
all_zscore_cols = [
    f'{m}_ZScore' for m in coop_metrics + mob_metrics if m not in ['Annual CO2 Emissions (Gigatons)']
] + [
    'Annual CO2 Emissions (Gigatons)_ZScore_Adjusted' # Assuming lower CO2 is better, so invert
] + [
    f'{m}_ZScore' for m in mob_metrics
] + [
    'Geopolitical Risk Index (GPR)_ZScore_Adjusted',
    'SPY Volatility (%)_ZScore_Adjusted'
]

# Correcting the CO2 Z-score adjustment (lower is better, so invert)
df['Annual CO2 Emissions (Gigatons)_ZScore_Adjusted'] = -df['Annual CO2 Emissions (Gigatons)_ZScore']

# Collect all Z-scores that need to be scaled (including inverted ones for consistency)
z_scores_to_scale = pd.concat([
    df[[f'{m}_ZScore' for m in coop_metrics if m != 'Annual CO2 Emissions (Gigatons)']],
    df['Annual CO2 Emissions (Gigatons)_ZScore_Adjusted'],
    df[[f'{m}_ZScore' for m in mob_metrics]],
    df['Geopolitical Risk Index (GPR)_ZScore_Adjusted'],
    df['SPY Volatility (%)_ZScore_Adjusted']
], axis=1)

# Determine the global min and max across all relevant Z-scores for scaling
min_original = z_scores_to_scale.min().min()
max_original = z_scores_to_scale.max().max()

# Define the new desired range
min_new = 0.25
max_new = 4.0

# Apply linear scaling to all individual Z-scores
scaled_z_scores = min_new + (z_scores_to_scale - min_original) * (max_new - min_new) / (max_original - min_original)

# Assign scaled Z-scores back to the DataFrame with new column names for clarity
for col in scaled_z_scores.columns:
    df[f'{col}_Scaled'] = scaled_z_scores[col]

# Calculate average scaled Z-scores for each category
df['Cooperation Z-Score'] = df[[f'{m}_ZScore_Scaled' for m in coop_metrics if m != 'Annual CO2 Emissions (Gigatons)']].mean(axis=1)
df['Cooperation Z-Score'] = df['Cooperation Z-Score'].add(df['Annual CO2 Emissions (Gigatons)_ZScore_Adjusted_Scaled'], axis=0) / 2 # Include adjusted CO2

df['Mobility Z-Score'] = df[[f'{m}_ZScore_Scaled' for m in mob_metrics]].mean(axis=1)
df['Predictability Z-Score'] = df[['Geopolitical Risk Index (GPR)_ZScore_Adjusted_Scaled', 'SPY Volatility (%)_ZScore_Adjusted_Scaled']].mean(axis=1)

# Select only the final average scaled Z-scores for further analysis
df = df[['Cooperation Z-Score', 'Mobility Z-Score', 'Predictability Z-Score']]

print("Calculated and Scaled Z-Scores (Average per Category, 0.25-4.0 range, higher is better):")
print(df)

```

    Libraries imported successfully.
    DataFrame created and 'Year' set as index.
    Calculated and Scaled Z-Scores (Average per Category, 0.25-4.0 range, higher is better):
          Cooperation Z-Score  Mobility Z-Score  Predictability Z-Score
    Year                                                               
    1975             1.877463          1.205031                1.891549
    1980             1.927945          1.307818                1.723588
    1985             2.077281          1.332866                1.858890
    1990             2.069560          1.526695                2.132491
    1995             2.082167          1.675422                2.369210
    2000             2.098878          1.830452                1.672366
    2005             2.038776          2.099967                2.759451
    2008             1.986921          2.136203                2.124929
    2010             1.966169          2.263998                2.487519
    2015             1.961315          2.481055                1.511524
    2020             1.930925          2.798041                1.966199
    2024             1.865427          3.225279                1.385111


## 2. Feature Engineering: Z-Score Deltas and GARCH Volatility

To enhance the predictive power of our LSTM model, we generate additional features:

-   **Z-Score Deltas**: The year-over-year change in each Z-score, capturing the rate of change.
-   **GARCH Volatility**: Estimates of conditional volatility using a GARCH(1,1) model on the Z-score deltas. GARCH models are effective in capturing volatility clustering (periods of high volatility followed by high volatility, and vice-versa), which is common in time series data, especially for 'Predictability'.


```python
# Calculate Z-Score deltas
for z_type in ['Mobility', 'Cooperation', 'Predictability']:
    df[f'{z_type} Z-Score Delta'] = df[f'{z_type} Z-Score'].diff().fillna(0)

# GARCH Volatility estimation
for z_type in ['Mobility', 'Cooperation', 'Predictability']:
    # arch_model requires a series with some variance. If a delta series is all zeros
    # (e.g., if Z-score didn't change), GARCH might fail. We assume some variation exists.
    # 'disp=off' suppresses verbose output during fitting.
    am = arch_model(df[f'{z_type} Z-Score Delta'], vol='Garch', p=1, q=1)
    res = am.fit(disp='off')
    df[f'{z_type} GARCH Volatility'] = res.conditional_volatility

# Display DataFrame with newly engineered features
print("DataFrame with Deltas and GARCH Volatility:")
print(df)

# Save checkpoint after GARCH calculation (optional, for debugging or later use)
df.to_csv('z_score_garch_checkpoint.csv', index=True)
print("\nCheckpoint saved to 'z_score_garch_checkpoint.csv'")
```

## 3. Data Preparation for LSTM

Long Short-Term Memory (LSTM) networks are designed to work with sequential data. For our time series forecasting, we need to prepare the data by:

1.  **Feature Selection**: Defining which columns will serve as inputs to the LSTM model.
2.  **Data Scaling**: Normalizing the features using `MinMaxScaler`. This is crucial for neural networks to ensure stable training and better performance.
3.  **Sequence Creation**: Transforming the flat time series into sequences (or windows) where each input sequence `X` consists of `look_back` past time steps, and the corresponding output `y` is the Z-score values at the next time step.


```python
# Define all features to be used in the LSTM model
features = ['Mobility Z-Score', 'Cooperation Z-Score', 'Predictability Z-Score',
            'Mobility Z-Score Delta', 'Cooperation Z-Score Delta', 'Predictability Z-Score Delta',
            'Mobility GARCH Volatility', 'Cooperation GARCH Volatility', 'Predictability GARCH Volatility']

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[features])

# Define the look-back period (number of previous time steps to consider for prediction)
look_back = 3

# Create input (X) and output (y) sequences for LSTM training
X, y = [], []
for i in range(len(scaled_data) - look_back):
    X.append(scaled_data[i:i + look_back])
    y.append(scaled_data[i + look_back, :3])  # Predicting only the three main Z-scores (first 3 features)

# Convert lists to NumPy arrays
X, y = np.array(X), np.array(y)

print(f"Shape of X (input sequences): {X.shape}") # Expected: (num_samples, look_back, num_features)
print(f"Shape of y (output targets): {y.shape}") # Expected: (num_samples, 3 Z-scores)

```

    Shape of X (input sequences): (9, 3, 9)
    Shape of y (output targets): (9, 3)


## 4. LSTM Model Definition and Training

We define a Sequential LSTM model with two `Dense` layers. The LSTM layer processes the sequential input, and the `Dense` layers map its output to our three predicted Z-scores. The model is compiled with the Adam optimizer and Mean Squared Error (MSE) loss. A `ModelCheckpoint` callback is used to save the best performing model (based on the lowest training loss) during the training process.


```python
# LSTM Model Architecture
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(look_back, len(features))))
model.add(Dense(32, activation='relu'))
model.add(Dense(3)) # Output layer for the 3 Z-scores (Mobility, Cooperation, Predictability)
model.compile(optimizer='adam', loss='mse')

model.summary()

# Checkpointing: Save the best model based on the lowest training loss
checkpoint = ModelCheckpoint('lstm_checkpoint.keras', save_best_only=True, monitor='loss', mode='min', verbose=1)

# Train the model
print("\nTraining LSTM Model...")
history = model.fit(X, y, epochs=100, batch_size=2, callbacks=[checkpoint], verbose=1)

print("\nLSTM Model Training Complete.")

# Optional: Plot training loss to visualize convergence
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.title('LSTM Model Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.show()

```





<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential_6"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ lstm_5 (<span style="color: #0087ff; text-decoration-color: #0087ff">LSTM</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)             │        <span style="color: #00af00; text-decoration-color: #00af00">18,944</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_10 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │         <span style="color: #00af00; text-decoration-color: #00af00">2,080</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_11 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)              │            <span style="color: #00af00; text-decoration-color: #00af00">99</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">21,123</span> (82.51 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">21,123</span> (82.51 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



    
    Training LSTM Model...
    Epoch 100/100
    [1m1/5[0m [32m━━━━[0m[37m━━━━━━━━━━━━━━━━[0m [1m0s[0m 7ms/step - loss: 4.0135e-05
    Epoch 100: loss did not improve from 0.00005
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 5.4830e-05
    
    LSTM Model Training Complete.



    
![png](loss.png)
    


## 5. Future Predictions

Using the trained LSTM model, we will now generate predictions for the next 20 years at 5-year intervals (2030, 2035, 2040, 2045). This is done iteratively:

1.  The model predicts the next step based on the last `look_back` historical data points.
2.  The predicted values (Z-scores) are then incorporated into the input sequence for the subsequent prediction, creating a rolling forecast.


```python
# Define the future years for which to make predictions
future_years = [2030, 2035, 2040, 2045]

# Initialize the last sequence with the most recent historical data for the first prediction
last_sequence = scaled_data[-look_back:].reshape(1, look_back, len(features))
future_preds_scaled = []

print("\nGenerating future predictions...")
for year in future_years:
    # Predict the next time step (3 Z-scores) in scaled format
    pred_scaled = model.predict(last_sequence, verbose=0)
    future_preds_scaled.append(pred_scaled.flatten()) # Store the scaled Z-score predictions

    # Prepare the next input sequence:
    # We take the predicted Z-scores and assume zeros for the delta and GARCH volatility features
    # for the predicted step. In a more complex model, these might also be predicted or derived.
    next_step_features_scaled = np.concatenate([pred_scaled.flatten(), np.zeros(len(features) - 3)])

    # Update the last_sequence by dropping the oldest step and adding the new predicted step
    last_sequence = np.concatenate([last_sequence[:, 1:, :], next_step_features_scaled.reshape(1, 1, -1)], axis=1)

# Convert list of scaled predictions to a NumPy array
future_preds_scaled_array = np.array(future_preds_scaled)

# Create a dummy array with the correct shape for inverse_transform. Only the first 3 columns
# (Z-scores) contain actual predictions; the rest are placeholders for inverse scaling.
dummy_for_inverse_transform = np.zeros((len(future_preds_scaled_array), len(features)))
dummy_for_inverse_transform[:, :3] = future_preds_scaled_array

# Inverse scale predictions to their original Z-score range
inverse_preds = scaler.inverse_transform(dummy_for_inverse_transform)[:, :3]

# Create a DataFrame for the future predictions
future_df = pd.DataFrame(inverse_preds, columns=['Mobility Z-Score', 'Cooperation Z-Score', 'Predictability Z-Score'])
future_df['Year'] = future_years
future_df.set_index('Year', inplace=True)

print("\nProjected Z-Scores (2030-2045):")
print(future_df)

```

    
    Generating future predictions...
    
    Projected Z-Scores (2030-2045):
          Mobility Z-Score  Cooperation Z-Score  Predictability Z-Score
    Year                                                               
    2030          3.163685             1.836391                1.627147
    2035          3.075196             1.836334                1.695471
    2040          2.722716             1.840445                1.769010
    2045          2.103065             1.866267                1.410011


## 6. Final Output and Visualization

Finally, we concatenate the historical and projected Z-scores into a single DataFrame. This combined DataFrame is then saved to a CSV file. A visualization is also provided to illustrate the historical trends and the future projections for each paradigm.


```python
import plotly.express as px

# Ensure 'Year' is a column
final_df = final_df.reset_index()

# Melt the DataFrame to long format for Plotly
melted_df = final_df.melt(
    id_vars='Year',
    value_vars=['Mobility Z-Score', 'Cooperation Z-Score', 'Predictability Z-Score'],
    var_name='Category',
    value_name='Z-Score'
)

# Create the Plotly line chart
fig = px.line(
    melted_df,
    x='Year',
    y='Z-Score',
    color='Category',
    markers=True,
    title='Historical and Projected Scores for Mobility, Cooperation, and Predictability'
)

# Add vertical line for projection
fig.add_vline(
    x=2025,
    line_dash='dash',
    line_color='gray',
    annotation_text='Start of Projection (2025)',
    annotation_position='top right'
)

# Layout customizations
fig.update_layout(
    xaxis_title='Year',
    yaxis_title='Z-Score',
    legend_title='Metric',
    template='plotly_white'
)

fig.show()

```



## References

1. **Bridgewater Associates**: The core concept and framework of "Modern Mercantilism" are attributed to Bridgewater Associates. While specific publications aren't listed in the provided LaTeX, these references indicate the foundational ideas come from Bridgewater's analysis.

2. **Heckscher, E.F.**: Refers to Eli F. Heckscher's work on *Mercantilism*, a seminal text on historical economic theory.

3. **Ekelund, R.B. Jr. & Tollison, R.D.**: Refers to the authors' work on mercantilism, often approached from a public choice perspective.

4. **Wilson, C.**: Likely refers to Charles Wilson’s historical analyses of mercantilism.

5. **Irwin, D.A.**: Refers to Douglas A. Irwin’s scholarship on trade policy and the historical treatment of mercantilism.

6. **Rodrik, D.**: Refers to Dani Rodrik’s work on globalization, trade, and industrial policy, especially discussions of "new mercantilism".

7. **World Trade Organization (WTO)**: General reference to WTO rules and their strategic enforcement in the context of modern trade.

8. **Dalio, R.**: Refers to Ray Dalio’s analysis, particularly his concept of "green, yellow, and red zones" in global economic/geopolitical landscapes.

9. **UNCTAD**:
    - [10]: UNCTAD’s work on digital services and cross-border data flows.
    - [38]: UNCTAD IIA Navigator, on trends in International Investment Agreement (IIA) terminations.

10. **World Bank**:
    - [12]: Data on Global Trade Openness and Recessions.
    - [13], [14]: Data on Trade (% of GDP).
    - [22]: International migrant stock.
    - [29]: Global financial openness metrics.

11. **Visual Capitalist**:
    - [16]: Historical internet usage and penetration.
    - [19]: Global internet traffic volume.

12. **ITU (International Telecommunication Union)**:
    - [17], [18]: Internet user statistics, penetration rates, and bandwidth usage.

13. **Our World in Data**:
    - [20]: UN DESA / International migrant stock.
    - [40], [41]: Global literacy rates.
    - [43]: Life expectancy data.
    - [51], [52]: Global CO2 emissions (Global Carbon Project).

14. **UN DESA**:
    - [21]: Data on international migrant stock.

15. **UN Migration Policies**:
    - [23]: Data or analysis on international migration policies.

16. **CBOE**:
    - [24]: Chicago Board Options Exchange - VIX index data.

17. **Investopedia**:
    - [25]: Definitions or data on the VIX index.
    - [31]: Definitions and examples of “black swan” events.

18. **Wikipedia**:
    - [26]: Information on the VIX index.
    - [32]: Lists or definitions of “black swan” events.

19. **FRED (Federal Reserve Economic Data)**:
    - [27]: Equity Market Volatility Tracker.

20. **IMF (International Monetary Fund)**:
    - [28]: Global Financial Stability Report.
    - [30]: Analysis of the 2009 global recession.

21. **GPR Index**:
    - [33], [34], [35]: News-based Geopolitical Risk Index developed by Matteo Iacoviello.

22. **BlackRock**:
    - [15]: BlackRock Geopolitical Risk Indicator (BGRI).

23. **UN Treaty Collection**:
    - [36]: Data on multilateral treaties.

24. **US Department of State**:
    - [37]: Information on international agreements entered by the U.S.

25. **Multilateralism Index**:
    - [23], [39]: Assesses participation/performance in multilateral institutions.

26. **UNESCO Institute for Statistics**:
    - [42]: Global literacy data.

27. **WHO (World Health Organization)**:
    - [43]: Global health indicators such as life expectancy.

28. **UNICEF**:
    - [46]: DTP3 vaccination coverage and zero-dose children data.

29. **UNFCCC**:
    - [49]: UN Framework Convention on Climate Change.

30. **Kyoto Protocol**:
    - [50]: Treaty committing states to reduce GHG emissions.

31. **Paris Agreement**:
    - [23]: Treaty on climate change.

32. **Global Health Expenditure**:
    - [47], [48]: Data on worldwide health spending.



