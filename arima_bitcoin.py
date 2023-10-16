import yfinance as yf
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import pickle

symbol = "BTC-USD"
period = "700d"
interval = "1h"
df_btc = yf.download(symbol, period=period, interval=interval)
df_btc = df_btc.resample('1H').ffill()
df_btc_close = df_btc['Close']

"""#Calculos por hora"""
"""#Entrenendo el Modelo ARIMA:"""
# Asegurarse de que los datos estén en formato de serie temporal
df_btc_close.index = pd.to_datetime(df_btc_close.index)

# Ajustar un modelo ARIMA
# model_pkl = ARIMA(df_btc_close, order=(5, 5, 5))  # Ejemplo con ARIMA(5,1,0)
# results = model_pkl.fit()

# Ruta para guardar el modelo y los resultados
# ruta_modelo = '/content/drive/MyDrive/MODELOS_ENTRENADOS/modelo_arima_pkls.pkl'

# Guardar el modelo y los resultados en un archivo
# with open(ruta_modelo, 'wb') as archivo_modelo:
#     pickle.dump({'model': model_pkl, 'results': results}, archivo_modelo)

"""#Extraccion del Modelo de la raiz:"""

# Ruta al archivo donde guardaste el modelo y los resultados
ruta_modelo = 'modelo_arima_pkls.pkl'

# Cargar el modelo y los resultados
with open(ruta_modelo, 'rb') as archivo_modelo:
    modelo_y_resultados = pickle.load(archivo_modelo)

# Recuperar el modelo y los resultados del diccionario
model_cargado = modelo_y_resultados['model']
results = modelo_y_resultados['results']

"""# Realizar predicciones"""

#Ultimo valor del BTC
df_last_close=df_btc['Close'].tail(3)
df_last_close=pd.DataFrame(df_last_close)
print('Ultimos datos de Close')
print(df_last_close)


# Realizar predicciones
forecast_steps = 10  # Por ejemplo, pronosticar 10 pasos en el futuro

# Realiza las predicciones
forecast = results.get_forecast(steps=forecast_steps)

# Extraer las predicciones y los intervalos de confianza
forecasted_values = forecast.predicted_mean
confidence_intervals = forecast.conf_int()

df_predicciones=pd.DataFrame(forecasted_values)
print('----------------------------------------------------------')
print('Predicciones')
print(df_predicciones)

# Graficar las predicciones y los datos originales
df_btc_close = df_btc['Adj Close']

# Realizar predicciones
forecast_steps = 5  # Por ejemplo, pronosticar 10 pasos en el futuro
forecast = results.get_forecast(steps=forecast_steps)

# Extraer las predicciones y los intervalos de confianza
forecasted_values = forecast.predicted_mean

# Crear forecast_dates y ajustar las etiquetas
last_date = df_btc_close.index[-1]
forecast_dates = pd.date_range(start=last_date, periods=forecast_steps, freq='H')  # Nota que eliminamos el paso adicional
date_labels = [date.strftime('%Y-%m-%d %H:%M') for date in forecast_dates]  # Incluye la hora y minutos

# Graficar las predicciones y los datos originales
ultimas_horas=100
plt.figure(figsize=(14, 7))
plt.plot(df_btc_close.tail(ultimas_horas), label='Datos Originales', color='blue')
plt.plot(forecast_dates, forecasted_values, label='Predicciones', color='red')
plt.fill_between(confidence_intervals.index, confidence_intervals.iloc[:, 0], confidence_intervals.iloc[:, 1], color='red', alpha=0.3, label='Intervalo de Confianza')

plt.legend()
plt.title('Predicciones de df_btc-USD con ARIMA por hora')

# Configurar las etiquetas del eje horizontal
plt.xticks(rotation=45, ha="right")

plt.show()



