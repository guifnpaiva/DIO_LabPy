# Projeto Covid Digital Innovation One

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima
from fbprophet import Prophet


def corrige_colunas(col_name) :
    return re.sub(r"[/| ]", "", col_name).lower()


def taxa_crescimento(data, variable, init_date=None, end_date=None) :
    if init_date is None :
        init_date = data.observationdate.loc[data[variable] > 0].min()
    else :
        init_date = pd.to_datetime(init_date)

    if end_date is None :
        end_date = data.observationdate.iloc[-1]
    else :
        end_date = pd.to_datetime(end_date)

    passado = data.loc[data.observationdate == init_date, variable].values[0]
    presente = data.loc[data.observationdate == end_date, variable].values[0]

    n = (end_date - init_date).days

    taxa = (presente / passado) ** (1 / n) - 1

    return taxa * 100


def taxa_crescimento_diaria(data, variable, init_date=None) :
    if init_date is None :
        init_date = data.observationdate.loc[data[variable] > 0].min()
    else :
        init_date = pd.to_datetime(init_date)

    end_date = data.observationdate.iloc[-1]
    passado = data.loc[data.observationdate == init_date, variable].values[0]
    presente = data.loc[data.observationdate == end_date, variable].values[0]

    n = (end_date - init_date).days

    taxas = list(map(
        lambda x : (data[variable].iloc[x] - data[variable].iloc[x - 1]) / data[variable].iloc[x - 1],
        range(1, n + 1)
    ))

    return np.array(taxas) * 100


# Importando os Dados
df = pd.read_csv("covid_19_data.csv", parse_dates=['ObservationDate', 'Last Update'])
print(df)

"""
Para visualizar os tipos de dados dm cada coluna
print(df.dtypes)
"""

# Eliminando os caracteres especiais dos nomes das colunas
df.columns = [corrige_colunas(col) for col in df.columns]

# Selecionando os dados de um País

"""
Para visualizar todos os países disponíveis
df.countryregion.unique
"""

brasil = df.loc[
    (df.countryregion == 'Brazil') &
    (df.confirmed > 0)
    ]

# Gráfico evolução gráficos confirmados
fig1 = px.line(brasil, 'observationdate', 'confirmed', title='Casos confirmados no Brasil')
fig1.show()

# Novos Casos por Dia
brasil['novoscasos'] = list(map(
    lambda x : 0 if (x == 0) else brasil['confirmed'].iloc[x] - brasil['confirmed'].iloc[x - 1],
    np.arange(brasil.shape[0])
))

fig2 = px.line(brasil, x='observationdate', y='novoscasos', title='Novos casos por dia',
               labels={'observationdate' : 'Data', 'novoscasos' : 'Novos casos'})
fig2.show()

# Mortes
fig3 = go.Figure()

fig3.add_trace(
    go.Scatter(x=brasil.observationdate, y=brasil.deaths, name='Mortes',
               mode='lines+markers', line={'color' : 'red'})
)

fig3.update_layout(title='Mortes por COVID-19 no Brasil')

fig3.show()

# Taxa de Crescimento
taxa_crescimento(brasil, 'confirmed')

# Taxa de Crescimento Diário
tx_dia = taxa_crescimento_diaria(brasil, 'confirmed')

primeiro_dia = brasil.observationdate.loc[brasil.confirmed > 0].min()

px.line(x=pd.date_range(primeiro_dia, brasil.observationdate.max())[1 :],
        y=tx_dia, title="Taxa de Crescimento de casos confirmados no Brasil")

# Predições
confirmados = brasil.confirmed
confirmados.index = brasil.observationdate
res = seasonal_decompose(confirmados)

fig4, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))

ax1.plot(res.observed)
ax2.plot(res.trend)
ax3.plot(res.seasonal)
ax4.plot(confirmados.index, res.resid)
ax4.axhline(0, linestyle='dashed', c='black')
plt.show()

# ARIMA
modelo = auto_arima(confirmados)
fig5 = go.Figure(go.Scatter(
    x=confirmados.index, y=confirmados, name='Observados'
))
fig5.add_trace(go.Scatter(
    x=confirmados.index, y=modelo.predict_in_sample(), name="Preditos"
))
fig5.add_trace(go.Scatter(
    x=pd.date_range('2020-05-20', "2020-06-20"), y=modelo.predict(61), name='Forecast'
))

fig5.update_layout(title='Previsão de casos confirmados no Brazil para os próximos 30 dias')

fig5.show()

# Modelo de Crescimento
train = confirmados.reset_index()[:-5]
test = confirmados.reset_index()[-5:]

# Renomear Colunas
train.rename(columns={"observationdate" : "ds", "confirmed" : "y"}, inplace=True)
test.rename(columns={"observationdate" : "ds", "confirmed" : "y"}, inplace=True)
test = test.set_index("ds")
test = test['y']

# Definir o modelo
profeta = Prophet(growth="logistic", changepoints=['2020-03-21', '2020-03-30', '2020-04-25',
                                                   '2020-05-03', '2020-05-10'])

pop = 211463256
train['cap'] = pop

# Treina o modelo
profeta.fit(train)

# Construindo previsões
future_dates = profeta.make_future_dataframe(periods=200)
future_dates['cap'] = pop
forecast = profeta.predict(future_dates)

fig6 = go.Figure()

fig6.add_trace(go.Scatter(x=forecast.ds, y=forecast.yhat, name='Predição'))
fig6.add_trace(go.Scatter(x=test.index, y=test, name='Observados - Teste'))
fig6.add_trace(go.Scatter(x=train.ds, y=train.y, name='Observados - Treino'))
fig6.update_layout(title='Predições de casos confirmados no Brasil')
fig6.show()