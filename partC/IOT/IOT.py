import os
import glob
import pandas as pd 
import plotly.graph_objects as go

DataDir = os.path.dirname(os.path.abspath('IOT.py')) + '/'
print(DataDir)

# files = os.listdir(DataDir)
# print(files)

# for root, dirs, files in os.walk(DataDir):
#   print("路徑：", root)
#   print("  目錄：", dirs)
#   print("  檔案：", files)

df = pd.read_csv('2019-10-27_20-37-27-Rest/Accelerometer.csv')
# Accelerometer
print(df)

fig = go.Figure(go.Scatter(x = df['Timestamp'], y = df['X'],name='Accelerometer'))

fig.update_layout(title='Accelerometer X when Study',
                   plot_bgcolor='rgb(230, 230,230)',
                   showlegend=True)

fig.show()
fig = go.Figure(go.Scatter(x = df['Timestamp'], y = df['Y'],name='Accelerometer'))

fig.update_layout(title='Accelerometer Y when Study',
                   plot_bgcolor='rgb(230, 230,230)',
                   showlegend=True)

fig.show()
fig = go.Figure(go.Scatter(x = df['Timestamp'], y = df['Z'],name='Accelerometer'))

fig.update_layout(title='Accelerometer Z when Study',
                   plot_bgcolor='rgb(230, 230,230)',
                   showlegend=True)

fig.show()

# Compass
df = pd.read_csv('2019-10-27_20-37-27-Rest/Compass.csv')

fig = go.Figure(go.Scatter(x = df['Timestamp'], y = df['X'],name='Compass'))

fig.update_layout(title='Compass X when Study',
                   plot_bgcolor='rgb(230, 230,230)',
                   showlegend=True)

fig.show()
fig = go.Figure(go.Scatter(x = df['Timestamp'], y = df['Y'],name='Compass'))

fig.update_layout(title='Compass Y when Study',
                   plot_bgcolor='rgb(230, 230,230)',
                   showlegend=True)

fig.show()
fig = go.Figure(go.Scatter(x = df['Timestamp'], y = df['Z'],name='Compass'))

fig.update_layout(title='Compass Z when Study',
                   plot_bgcolor='rgb(230, 230,230)',
                   showlegend=True)

fig.show()

# GPS
df = pd.read_csv('2019-10-27_20-37-27-Rest/GPS.csv')

fig = go.Figure(go.Scatter(x = df['Timestamp'], y = df['Latitude'],name='GPS'))

fig.update_layout(title='GPS Latitude when Study',
                   plot_bgcolor='rgb(230, 230,230)',
                   showlegend=True)

fig.show()
fig = go.Figure(go.Scatter(x = df['Timestamp'], y = df['Longitude'],name='GPS'))

fig.update_layout(title='GPS Longitude when Study',
                   plot_bgcolor='rgb(230, 230,230)',
                   showlegend=True)

fig.show()
fig = go.Figure(go.Scatter(x = df['Timestamp'], y = df['Altitude'],name='GPS'))

fig.update_layout(title='GPS Altitude when Study',
                   plot_bgcolor='rgb(230, 230,230)',
                   showlegend=True)

fig.show()

