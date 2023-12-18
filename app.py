import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv('connections.csv')
df['connected_at'] = pd.to_datetime(df['connected_at'])

# Create a list of unique employees
employees = df['account_owner'].unique().tolist()
employees.insert(0, 'All')

st.title("CLS 2023 Year in Review")
st.markdown("Since April 2023 LinkedIn has become ripe with opportunities. The graph below showcases weekly connections. Remarkable YoY growth stats; Suzie a 284% increase, Chris a 360% increase, Sharon a 155% increase, Sam a 287% increase with 7,431 total new connections added overall")
st.markdown("As we look to 2024, even with conservative estimates (using the previous 6 month av. weekly connections) each account should more than double again.")

col1, col2, col3 = st.columns(3)
with col1:
    selected_employee = st.selectbox('Select Account', employees)
with col2:
    pass
with col3:
    start_date, end_date = st.date_input('Date Range', [datetime(2023, 1, 1), datetime(2023, 12, 31)])
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Filter data based on user input
if selected_employee == 'All':
    filtered_data = df[df['connected_at'].between(start_date, end_date)]
else:
    filtered_data = df[(df['connected_at'].between(start_date, end_date)) & (df['account_owner'] == selected_employee)]
    
# Group data by week and plot
weekly_data = filtered_data.groupby(['connected_at', 'account_owner']).count().unstack().resample('W').sum()

tab1, tab2 = st.tabs(["Graph", "Data"])

with tab1:
    st.bar_chart(weekly_data['connections'])

with tab2:
    # Drop the 'organization_start_1' column
    filtered_data = filtered_data.drop(columns=['organization_start_1'])
    # Convert 'connected_at' to date only (no time)
    filtered_data['connected_at'] = filtered_data['connected_at'].dt.date
    # Display the dataframe without index
    st.dataframe(filtered_data, hide_index=True)

# Calculate trajectory of new connections
before_april = df[df['connected_at'] < datetime(2023, 4, 1)]
after_april = df[df['connected_at'] >= datetime(2023, 4, 1)]

# Forecasting model
model = LinearRegression()
X = np.array(range(len(after_april))).reshape(-1, 1)
y = after_april['connections'].values

# Check if y contains NaN values and handle them
if np.isnan(y).any():
    y = np.nan_to_num(y)

model.fit(X, y)
forecast = model.predict(np.array([len(after_april) + i for i in range(53)]).reshape(-1, 1))
summary_data = []
for employee in employees:
    if employee == 'All':
        continue
    current_total = df[df['account_owner'] == employee].shape[0]
    new_since_april = after_april[after_april['account_owner'] == employee].shape[0]
    starting_connections = current_total - new_since_april
    e2024_est = current_total + (forecast[-1])
    percent_increase = (new_since_april / starting_connections) * 100 if starting_connections != 0 else 0
    summary_data.append({'Employee': employee, 'Starting Connections': starting_connections, 'New Since April': new_since_april, 'Current Total': current_total, '% Increase': round(percent_increase), 'E2024 Est.': round(e2024_est)})

# Add a Totals row at the bottom of the data frame with a SUM of all employees for each column
summary_data.append({'Employee': 'Total', 'Starting Connections': sum([data['Starting Connections'] for data in summary_data]), 'New Since April': sum([data['New Since April'] for data in summary_data]), 'Current Total': sum([data['Current Total'] for data in summary_data]), '% Increase': '', 'E2024 Est.': sum([data['E2024 Est.'] for data in summary_data])})

summary = pd.DataFrame(summary_data)
summary['% Increase'] = summary['% Increase'].apply(lambda x: f'{x:.0f}%' if x != '' else '')
st.dataframe(summary, hide_index=True)
