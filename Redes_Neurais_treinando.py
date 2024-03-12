from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from iqoptionapi.stable_api import IQ_Option
import time 
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import BatchNormalization
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import LSTM 
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.trend import PSARIndicator
from keras.models import Sequential

short_window = 40
long_window = 10
model_inputs = []

# Conectando à IQ Option
I_want_money = IQ_Option("E-mail", "Senha")
I_want_money.connect()

# Verificando se a conexão foi bem-sucedida
if I_want_money.check_connect():
    print('Conexão bem-sucedida!')
 # Solicita ao usuário que escolha entre a conta demo ou real
    conta_tipo = input("Por favor, escolha o tipo de conta (demo/real): ").strip().lower()

    if conta_tipo == 'demo':
        I_want_money.change_balance('PRACTICE')  # Muda para a conta demo
        print("Você escolheu a conta demo.")
    elif conta_tipo == 'real':
        I_want_money.change_balance('REAL')  # Muda para a conta real
        print("Você escolheu a conta real.")
    else:
        print("Entrada inválida. Por favor, reinicie e escolha 'demo' ou 'real'.")
        exit()  # Encerra o script se a entrada for inválida

else:
    print('Erro na conexão.')
    I_want_money.connect()
    

def iniciar_operacao():
    print("Bem-vindo! Iniciando as operações do robô...\n")
    
    
    # Simulando algumas etapas de operação com prints
    print("Etapa 1: Calibrando sensores...")
    # Aqui você colocaria o código para calibrar os sensores
    
    
    print("Calibração concluída.\n")
    

    print("Etapa 2: Verificando a conexão com o sistema de controle...")
    # Código para verificar a conexão
    
    print("Conexão estabelecida.\n")
    

    print("Etapa 3: Iniciando a sequência de operações principais...")
    # Código para iniciar operações principais
    
    print("Operações principais em andamento...\n")
   

    # Após completar todas as etapas necessárias
    print("Todas as operações foram concluídas com sucesso. Aguardando novas instruções.")
    
# Chamando a função para iniciar o processo
iniciar_operacao()

time.sleep(0.1)
    
# Definição das funções de compra e venda
def comprar(par, valor, tempo):
    I_want_money.buy(valor, par, "call", tempo)

def vender(par, valor, tempo):
    I_want_money.buy(valor, par, "put", tempo)     
  # Tentativa de carregar modelo anterior
try:
    model_1 = load_model("meu_modelo_1.keras")
    print("Modelo 1 carregado com sucesso!")
except:
    model_1 = None


     # Loop de aprendizado contínuo (aqui, usando um for como exemplo para 30 dias)
for day in range(1):
    print(f"Tomando decisão para o dia  {day+1}")
    

 # Coletando dados dos pares de moedas 
timeframe = 60 * 1 
end_from_time = time.time()
two_years_in_seconds = 60 * 60 * 24 * 365 * 12

pairs = ['EURUSD']
all_data = []

for pair in pairs:
    data = I_want_money.get_candles(pair, timeframe, int((end_from_time - two_years_in_seconds) / timeframe), end_from_time)
    all_data.append(pd.DataFrame(data))
   

df = pd.concat(all_data, ignore_index=True)
    

     # Convertendo a coluna de tempo para o formato datetime
df['time'] = pd.to_datetime(df['from'], unit='s')
    
     # Implementação de características adicionais
df['media_movel_20'] = df['close'].rolling(window=20).mean()
df['banda_bollinger_superior'], df['banda_bollinger_inferior'] = df['close'].rolling(window=20).mean() + 2*df['close'].rolling(window=20).std(), df['close'].rolling(window=20).mean() - 2*df['close'].rolling(window=20).std()
df['banda_bollinger_superior'], df['banda_bollinger_inferior'] = df['close'].rolling(window=20).mean() + 2*df['close'].rolling(window=20).std(), df['close'].rolling(window=20).mean() - 2*df['close'].rolling(window=20).std()
    
     # Identificação de divergências para as Bandas de Bollinger
df['bollinger_divergencia'] = 0
df.loc[(df['close'] > df['close'].shift(1)) & (df['close'] < df['banda_bollinger_inferior'].shift(1)), 'bollinger_divergencia'] = 1  # Divergência positiva
df.loc[(df['close'] < df['close'].shift(1)) & (df['close'] > df['banda_bollinger_superior'].shift(1)), 'bollinger_divergencia'] = -1  # Divergência negativa
    
bollinger_divergencia_positiva = df[df['bollinger_divergencia'] == 1]
bollinger_divergencia_negativa = df[df['bollinger_divergencia'] == -1]

if not bollinger_divergencia_positiva.empty:
    print("Divergência positiva de Bollinger identificada nas seguintes datas:")
    print(bollinger_divergencia_positiva.index)

if not bollinger_divergencia_negativa.empty:
    print("\nDivergência negativa de Bollinger identificada nas seguintes datas:")
    print(bollinger_divergencia_negativa.index)
        
     # Cálculo do RSI
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).fillna(0)
loss = (-delta.where(delta < 0, 0)).fillna(0)

avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()

rs = avg_gain / avg_loss
df['rsi'] = 100 - (100 / (1 + rs))

     # Identificação de divergências RSI
df['rsi_divergencia'] = 0  # Cria uma nova coluna com valores padrão 0
df.loc[(df['close'] > df['close'].shift(1)) & (df['rsi'] < df['rsi'].shift(1)), 'rsi_divergencia'] = 1  # Divergência positiva
df.loc[(df['close'] < df['close'].shift(1)) & (df['rsi'] > df['rsi'].shift(1)), 'rsi_divergencia'] = -1  # Divergência negativa

rsi_divergencia_positiva = df[df['rsi_divergencia'] == 1]
rsi_divergencia_negativa = df[df['rsi_divergencia'] == -1]

if not rsi_divergencia_positiva.empty:
    print("Divergência positiva de RSI identificada nas seguintes datas:")
    print(rsi_divergencia_positiva.index)

if not rsi_divergencia_negativa.empty:
    print("\nDivergência negativa de RSI identificada nas seguintes datas:")
    print(rsi_divergencia_negativa.index)
        
     # Implementação do indicador MACD
macd = MACD(df['close'])
df['macd'] = macd.macd()
df['macd_signal'] = macd.macd_signal()
df['macd_diff'] = macd.macd_diff()
    
     # Identificação de divergências MACD
df['macd_divergencia'] = 0  # Cria uma nova coluna com valores padrão 0
df.loc[(df['close'] > df['close'].shift(1)) & (df['macd'] < df['macd'].shift(1)), 'macd_divergencia'] = 1  # Divergência positiva
df.loc[(df['close'] < df['close'].shift(1)) & (df['macd'] > df['macd'].shift(1)), 'macd_divergencia'] = -1  # Divergência negativa

macd_divergencia_positiva = df[df['macd_divergencia'] == 1]
macd_divergencia_negativa = df[df['macd_divergencia'] == -1]

if not macd_divergencia_positiva.empty:
    print("Divergência positiva de MACD identificada nas seguintes datas:")
    print(macd_divergencia_positiva.index)

if not macd_divergencia_negativa.empty:
    print("\nDivergência negativa de MACD identificada nas seguintes datas:")
    print(macd_divergencia_negativa.index)
        
     # Definição do período para o Canal de Donchian
periodo_donchian = 20

     # Cálculo do Canal de Donchian
df['donchian_high'] = df['max'].rolling(window=periodo_donchian).max()
df['donchian_low'] = df['min'].rolling(window=periodo_donchian).min() 
    
     # Identificação de divergências para o Canal de Donchian
df['donchian_divergencia'] = 0
      # Divergência positiva
df.loc[(df['close'].shift(1) < df['donchian_low'].shift(1)) & (df['close'] > df['donchian_low']), 'donchian_divergencia'] = 1  
     # Divergência negativa
df.loc[(df['close'].shift(1) > df['donchian_high'].shift(1)) & (df['close'] < df['donchian_high']), 'donchian_divergencia'] = -1  

      # Imprimir pontos de divergência identificados
donchian_divergencia_positiva = df[df['donchian_divergencia'] == 1]
donchian_divergencia_negativa = df[df['donchian_divergencia'] == -1]

if not donchian_divergencia_positiva.empty:
    print("Divergência positiva do Canal de Donchian identificada nas seguintes datas:")
    print(donchian_divergencia_positiva.index)

if not donchian_divergencia_negativa.empty:
    print("\nDivergência negativa do Canal de Donchian identificada nas seguintes datas:")
    print(donchian_divergencia_negativa.index)  
     
     #SAR Parabolic   
indicator_psar = PSARIndicator(df['max'], df['min'], df['close'], step=0.02, max_step=0.2)
df['psar'] = indicator_psar.psar() 
    
df['psar_divergencia'] = 0
     # Divergência positiva
df.loc[(df['psar'] > df['close']) & (df['close'] > df['close'].shift(1)), 'psar_divergencia'] = 1
     # Divergência negativa
df.loc[(df['psar'] < df['close']) & (df['close'] < df['close'].shift(1)), 'psar_divergencia'] = -1

     # Imprimir pontos de divergência identificados
psar_divergencia_positiva = df[df['psar_divergencia'] == 1]
psar_divergencia_negativa = df[df['psar_divergencia'] == -1]

if not psar_divergencia_positiva.empty:
    print("Divergência positiva do SAR Parabólico identificada nas seguintes datas:")
    print(psar_divergencia_positiva.index)

if not psar_divergencia_negativa.empty:
    print("\nDivergência negativa do SAR Parabólico identificada nas seguintes datas:")
    print(psar_divergencia_negativa.index)    
    
    # Definindo um limite para considerar abertura e fechamento como iguais
limite_doji = 0.001

df['doji'] = np.where(abs(df['close'] - df['open']) <= limite_doji, 1, 0)   
    
df['doji_divergencia'] = 0
      # Divergência positiva
df.loc[(df['doji'] == 1) & (df['close'].shift(1) < df['close'].shift(2)) & (df['close'] > df['open']), 'doji_divergencia'] = 1
     # Divergência negativa
df.loc[(df['doji'] == 1) & (df['close'].shift(1) > df['close'].shift(2)) & (df['close'] < df['open']), 'doji_divergencia'] = -1

     # Imprimir pontos de divergência identificados
doji_divergencia_positiva = df[df['doji_divergencia'] == 1]
doji_divergencia_negativa = df[df['doji_divergencia'] == -1]

if not doji_divergencia_positiva.empty:
    print("Divergência positiva do Doji identificada nas seguintes datas:")
    print(doji_divergencia_positiva.index)

if not doji_divergencia_negativa.empty:
    print("\nDivergência negativa do Doji identificada nas seguintes datas:")
    print(doji_divergencia_negativa.index)
         
     # Cálculo das velas Heikin-Ashi
df['ha_close'] = (df['open'] + df['max'] + df['min'] + df['close']) / 4

ha_open = [(df['open'][0] + df['close'][0]) / 2]
for i in range(1, len(df)):
    ha_open.append((ha_open[i-1] + df['ha_close'][i-1]) / 2)
df['ha_open'] = ha_open

df['ha_high'] = df[['max', 'ha_open', 'ha_close']].max(axis=1)
df['ha_low'] = df[['min', 'ha_open', 'ha_close']].min(axis=1)
    
     # Identificação de divergências para Heikin-Ashi
df['ha_divergencia'] = 0
     # Divergência positiva
df.loc[(df['ha_close'] > df['ha_open']) & (df['close'] < df['close'].shift(1)), 'ha_divergencia'] = 1  
     # Divergência negativa
df.loc[(df['ha_close'] < df['ha_open']) & (df['close'] > df['close'].shift(1)), 'ha_divergencia'] = -1  

     #  Imprimir pontos de divergência identificados
ha_divergencia_positiva = df[df['ha_divergencia'] == 1]
ha_divergencia_negativa = df[df['ha_divergencia'] == -1]

if not ha_divergencia_positiva.empty:
    print("Divergência positiva de Heikin-Ashi identificada nas seguintes datas:")
    print(ha_divergencia_positiva.index)

if not ha_divergencia_negativa.empty:
    print("\nDivergência negativa de Heikin-Ashi identificada nas seguintes datas:")
    print(ha_divergencia_negativa.index)
    
     # Cálculo de Médias Móveis
df['media_movel_curta'] = df['close'].rolling(window=5).mean()
df['media_movel_longa'] = df['close'].rolling(window=20).mean()
    
     # Identificação de Convergências
df['convergencia'] = 0  # Cria uma nova coluna com valores padrão 0
df.loc[df['media_movel_curta'] > df['media_movel_longa'], 'convergencia'] = 1  # Marca pontos de convergência positiva com 1
df.loc[df['media_movel_curta'] < df['media_movel_longa'], 'convergencia'] = -1  # Marca pontos de convergência negativa com -1

       # Verificando e imprimindo pontos de convergência
convergencias_previas = df['convergencia'].shift(1)
for index, (convergencia_atual, convergencia_anterior) in enumerate(zip(df['convergencia'], convergencias_previas)):
    if pd.notnull(convergencia_anterior):  # Ignora a primeira linha onde não há valor anterior
        if convergencia_atual == 1 and convergencia_anterior != 1:
            print(f"Convergência positiva no índice {index}")
        elif convergencia_atual == -1 and convergencia_anterior != -1:
            print(f"Convergência negativa no índice {index}")

def calculate_adx(df, window=14):
      # Calculando as variações positivas e negativas entre períodos
    df['DMp'] = np.where(((df['max'] - df['max'].shift(1)) > (df['min'].shift(1) - df['min'])) & 
                         ((df['max'] - df['max'].shift(1)) > 0), df['max'] - df['max'].shift(1), 0)
    df['DMn'] = np.where(((df['min'].shift(1) - df['min']) > (df['max'] - df['max'].shift(1))) & 
                         ((df['min'].shift(1) - df['min']) > 0), df['min'].shift(1) - df['min'], 0)

    # Calculando TR
    df['TR'] = np.maximum(df['max'] - df['min'], 
                          np.maximum(abs(df['max'] - df['close'].shift(1)), 
                                     abs(df['min'] - df['close'].shift(1))))

    # Calculando médias móveis verdadeiras
    df['TRm'] = df['TR'].rolling(window=window).mean()
    df['DMpm'] = df['DMp'].rolling(window=window).mean()
    df['DMnm'] = df['DMn'].rolling(window=window).mean()

    # Calculando DIp e DIn
    df['DIp'] = (df['DMpm'] / df['TRm']) * 100
    df['DIn'] = (df['DMnm'] / df['TRm']) * 100

    # Calculando DX
    df['DX'] = (abs(df['DIp'] - df['DIn']) / (df['DIp'] + df['DIn'])) * 100

    # Calculando ADX
    df['ADX'] = df['DX'].rolling(window=window).mean()
    
    return df

# Chamar a função para calcular ADX
df = calculate_adx(df)

# Identificação de Divergências de Alta (Bullish) no ADX
df['Bullish_Divergence'] = np.where((df['ADX'] > df['ADX'].shift(1)) & (df['close'] < df['close'].shift(1)), 1, 0)

# Identificação de Divergências de Baixa (Bearish) no ADX
df['Bearish_Divergence'] = np.where((df['ADX'] < df['ADX'].shift(1)) & (df['close'] > df['close'].shift(1)), 1, 0)


def calculate_atr(df, window=14):
    # Calculando TR
    df['TR'] = np.maximum(df['max'] - df['min'], 
                          np.maximum(abs(df['max'] - df['close'].shift(1)), 
                                     abs(df['min'] - df['close'].shift(1))))

    # Calculando ATR
    df['ATR'] = df['TR'].rolling(window=window).mean()
    
    return df

# Chamar a função para calcular ATR
df = calculate_atr(df)

# Identificação de Divergências de Alta (Bullish) no ATR
df['Bullish_ATR_Divergence'] = np.where((df['ATR'] > df['ATR'].shift(1)) & (df['close'] < df['close'].shift(1)), 1, 0)

# Identificação de Divergências de Baixa (Bearish) no ATR
df['Bearish_ATR_Divergence'] = np.where((df['ATR'] < df['ATR'].shift(1)) & (df['close'] > df['close'].shift(1)), 1, 0)


# Cálculo do Indicador Envelopes
envelopes_period = 20  # Escolha o período para o indicador Envelopes
envelopes_percentage = 0.03  # Escolha a porcentagem para o desvio do envelope (3% é comum)

# Calculando a Média Móvel Simples (SMA)
df['SMA'] = df['close'].rolling(window=envelopes_period).mean()

# Calculando os Limites do Envelope Superior e Inferior
df['Upper_Envelope'] = df['SMA'] * (1 + envelopes_percentage)
df['Lower_Envelope'] = df['SMA'] * (1 - envelopes_percentage)

# Identificação de Divergências no Envelope Superior
df['Bullish_Envelope_Divergence'] = np.where(
    (df['Upper_Envelope'] > df['Upper_Envelope'].shift(1)) & (df['close'] < df['close'].shift(1)), 1, 0)

df['Bearish_Envelope_Divergence'] = np.where(
    (df['Upper_Envelope'] < df['Upper_Envelope'].shift(1)) & (df['close'] > df['close'].shift(1)), 1, 0)


def KST(df):
    """
    Calcula o indicador Know Sure Thing (KST) para o DataFrame fornecido.
    """
    # Definição dos parâmetros
    r1, r2, r3, r4 = 10, 15, 20, 30
    n1, n2, n3, n4, n_sig = 10, 10, 10, 15, 9

    # Calculando taxas de mudança
    M = df['close'].diff(r1 - 1)
    N = df['close'].shift(r1 - 1)
    ROC1 = M / N

    M = df['close'].diff(r2 - 1)
    N = df['close'].shift(r2 - 1)
    ROC2 = M / N

    M = df['close'].diff(r3 - 1)
    N = df['close'].shift(r3 - 1)
    ROC3 = M / N

    M = df['close'].diff(r4 - 1)
    N = df['close'].shift(r4 - 1)
    ROC4 = M / N

    # Definindo o KST
    KST = pd.Series(ROC1.rolling(window=n1).sum() + ROC2.rolling(window=n2).sum() * 2 + ROC3.rolling(window=n3).sum() * 3 + ROC4.rolling(window=n4).sum() * 4, name='KST')
    df['KST_Signal'] = KST.rolling(window=n_sig).mean()

    df['KST'] = KST

    return df

 # Para usar a função, simplesmente chame:
df = KST(df)

 # Divergências Bullish e Bearish
df['Bullish_KST_Divergence'] = np.where((df['close'] > df['close'].shift(1)) & (df['KST'] < df['KST'].shift(1)), 1, 0)
df['Bearish_KST_Divergence'] = np.where((df['close'] < df['close'].shift(1)) & (df['KST'] > df['KST'].shift(1)), 1, 0)


def add_chaos_fractals_to_df(df):
    df['Max_Fractal'] = 0
    df['Min_Fractal'] = 0
    
    for i in range(2, len(df)-2):
        if df['max'].iloc[i] == max(df['max'].iloc[i-2:i+3]):
            df['Max_Fractal'].iloc[i] = 1
        if df['min'].iloc[i] == min(df['min'].iloc[i-2:i+3]):
            df['Min_Fractal'].iloc[i] = 1
            
    return df

def add_rsi_divergence_to_df(df, window=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['RSI_Bullish_Divergence'] = (df['close'] < df['close'].rolling(window=window).min()) & (df['rsi'] > df['rsi'].rolling(window=window).max())
    df['RSI_Bearish_Divergence'] = (df['close'] > df['close'].rolling(window=window).max()) & (df['rsi'] < df['rsi'].rolling(window=window).min())
    
    return df

df = add_chaos_fractals_to_df(df)
df = add_rsi_divergence_to_df(df)

# Calcular as médias móveis
df['SMA_9'] = df['close'].rolling(window=9).mean()
df['SMA_6'] = df['close'].rolling(window=6).mean()

# Calcular a divergência entre as médias móveis
df['SMA_Divergence'] = df['SMA_6'] - df['SMA_9']

# Identificar os cruzamentos bullish e bearish
df['SMA_Bullish_Crossover'] = ((df['SMA_6'] > df['SMA_9']) & (df['SMA_6'].shift(1) <= df['SMA_9'].shift(1))).astype(int)
df['SMA_Bearish_Crossover'] = ((df['SMA_6'] < df['SMA_9']) & (df['SMA_6'].shift(1) >= df['SMA_9'].shift(1))).astype(int)


   
      # Removendo linhas com valores NaN
df = df.dropna()

        # Coluna 'target'
df['target'] = df['close'].diff().apply(lambda x: 1 if x > 0 else 0)


        # Especificando as colunas dos indicadores no DataFrame para serem usadas como features
print(df.columns)
X = df[['close', 'open', 'max', 'min', 'media_movel_20', 'banda_bollinger_superior', 'banda_bollinger_inferior', 'bollinger_divergencia', 'ha_open', 'ha_close', 'ha_high', 'ha_low', 'ha_divergencia', 'media_movel_curta', 'media_movel_longa', 'convergencia', 'macd', 'macd_divergencia', 'rsi', 'rsi_divergencia', 'donchian_high', 'donchian_low', 'donchian_divergencia', 'psar', 'psar_divergencia', 'doji', 'doji_divergencia', 'ADX', 'Bullish_Divergence', 'Bearish_Divergence', 'ATR', 'Bullish_ATR_Divergence', 'Bearish_ATR_Divergence', 'SMA', 'Upper_Envelope', 'Lower_Envelope', 'Bullish_Envelope_Divergence', 'Bearish_Envelope_Divergence', 'KST', 'KST_Signal', 'Bullish_KST_Divergence', 'Bearish_KST_Divergence', 'Max_Fractal', 'Min_Fractal', 'RSI_Bullish_Divergence', 'RSI_Bearish_Divergence', 'SMA_9', 'SMA_6', 'SMA_Divergence', 'SMA_Bullish_Crossover', 'SMA_Bearish_Crossover']]
y = df['target']


       # Dividindo os dados para treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalizando os dados
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

         # Remodelando os dados para LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_train.shape[1], 1)
print(X_train.shape)
print(X_test.shape)
    
   # Definindo a Primeira Rede LSTM (RN1)
model_1 = Sequential()
model_1.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(51, 1)))
model_1.add(Dropout(0.2))
model_1.add(LSTM(25, activation='relu', return_sequences=True))  # Camada LSTM extra
model_1.add(Dropout(0.2))
model_1.add(Dense(1, activation='sigmoid'))

optimizer_1 = RMSprop(learning_rate=0.001)
model_1.compile(optimizer=optimizer_1, loss='binary_crossentropy', metrics=['accuracy'])
    
     # Implementação de callbacks
lr_adjuster_1 = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_lr=0.00001)
early_stopper_1 = EarlyStopping(patience=5)


 # Treinando o Model_1
print("Começando o treinamento do Model_1.")
model_1.fit(X_train, y_train, epochs=2500, batch_size=128)  
print("Treinamento do Model_1 concluído.")

     # Salvar o Model_1
model_1.save("meu_modelo_1.keras")
print(f"Modelo 1 atualizado e salvo.")


      
while True:
  
    
    indices_de_erros = []
    
        # Previsões do modelo
    predictions_1 = model_1.predict(X_test)
    
    start_index = 0
    
    for i in range(start_index, len(df)):
    
         # Condições de Compra
        condicao_compra_1 = df['media_movel_curta'].iloc[i] > df['media_movel_longa'].iloc[i]
        condicao_compra_2 = df['rsi'].iloc[i] < 30
        condicao_compra_3 = df['macd'].iloc[i] > df['macd_signal'].iloc[i]
        condicao_compra_4 = df['doji_divergencia'].iloc[i] == 1
        condicao_compra_5 = df['psar_divergencia'].iloc[i] == 1
        condicao_compra_6 = df['donchian_divergencia'].iloc[i] == 1
        condicao_compra_7 = df['ha_divergencia'].iloc[i] == 1
        condicao_compra_8 = df['bollinger_divergencia'].iloc[i] == 1
        condicao_compra_9 = df['ADX'].iloc[i] > df['ADX'].shift(1).iloc[i]
        condicao_compra_10 = df['Bullish_Divergence'].iloc[i] == 1
        condicao_compra_11 = df['Bullish_ATR_Divergence'].iloc[i] == 1
        condicao_compra_12 = df['ATR'].iloc[i] > df['ATR'].shift(1).iloc[i]
        condicao_compra_13 = df['close'].iloc[i] < df['Lower_Envelope'].iloc[i]
        condicao_compra_14 = df['Bullish_Envelope_Divergence'].iloc[i] == 1
        condicao_compra_15 = (df['KST'].iloc[i] > df['KST_Signal'].iloc[i]) & (df['KST'].shift(1).iloc[i] <= df['KST_Signal'].shift(1).iloc[i])
        condicao_compra_16 = df['Bullish_KST_Divergence'].iloc[i] == 1
        condicao_compra_17 = df['Max_Fractal'].iloc[i] == 1
        condicao_compra_18 = df['RSI_Bullish_Divergence'].iloc[i] == 1
        condicao_compra_19 = df['SMA_6'].iloc[i] > df['SMA_9'].iloc[i]
        condicao_compra_20 = (df['SMA_6'].iloc[i] > df['SMA_9'].iloc[i]) & (df['SMA_6'].shift(1).iloc[i] <= df['SMA_9'].shift(1).iloc[i])
        print(f"Condições de compra: {[condicao_compra_1, condicao_compra_2, condicao_compra_3, condicao_compra_4, condicao_compra_5, condicao_compra_6, condicao_compra_7, condicao_compra_8, condicao_compra_9, condicao_compra_10, condicao_compra_11, condicao_compra_12, condicao_compra_13, condicao_compra_14, condicao_compra_15, condicao_compra_16, condicao_compra_17, condicao_compra_18, condicao_compra_19, condicao_compra_20]}")
        
        
         # Condições de Venda
        condicao_venda_1 = df['media_movel_curta'].iloc[i] < df['media_movel_longa'].iloc[i]
        condicao_venda_2 = df['rsi'].iloc[i] > 70
        condicao_venda_3 = df['macd'].iloc[i] < df['macd_signal'].iloc[i]
        condicao_venda_4 = df['doji_divergencia'].iloc[i] == -1
        condicao_venda_5 = df['psar_divergencia'].iloc[i] == -1
        condicao_venda_6 = df['donchian_divergencia'].iloc[i] == -1
        condicao_venda_7 = df['ha_divergencia'].iloc[i] == -1
        condicao_venda_8 = df['bollinger_divergencia'].iloc[i] == -1
        condicao_venda_9 = df['ADX'].iloc[i] < df['ADX'].shift(1).iloc[i]
        condicao_venda_10 = df['Bearish_Divergence'].iloc[i] == 1
        condicao_venda_11 = df['Bearish_ATR_Divergence'].iloc[i] == 1
        condicao_venda_12 = df['ATR'].iloc[i] < df['ATR'].shift(1).iloc[i]
        condicao_venda_13 = df['close'].iloc[i] > df['Upper_Envelope'].iloc[i]
        condicao_venda_14 = df['Bearish_Envelope_Divergence'].iloc[i] == 1
        condicao_venda_15 = (df['KST'].iloc[i] < df['KST_Signal'].iloc[i]) & (df['KST'].shift(1).iloc[i] >= df['KST_Signal'].shift(1).iloc[i])
        condicao_venda_16 = df['Bearish_KST_Divergence'].iloc[i] == 1
        condicao_venda_17 = df['Min_Fractal'].iloc[i] == 1
        condicao_venda_18 = df['RSI_Bearish_Divergence'].iloc[i] == 1
        condicao_venda_19 = df['SMA_6'].iloc[i] < df['SMA_9'].iloc[i]
        condicao_venda_20 = (df['SMA_6'].iloc[i] < df['SMA_9'].iloc[i]) & (df['SMA_6'].shift(1).iloc[i] >= df['SMA_9'].shift(1).iloc[i])
        print(f"Condições de venda: {[condicao_venda_1, condicao_venda_2, condicao_venda_3, condicao_venda_4, condicao_venda_5, condicao_venda_6, condicao_venda_7, condicao_venda_8, condicao_venda_9, condicao_venda_10, condicao_venda_11, condicao_venda_12, condicao_venda_13, condicao_venda_14, condicao_venda_15, condicao_venda_16, condicao_venda_17, condicao_venda_18,  condicao_venda_19,  condicao_venda_20]}")
        
           # Verifica se duas ou mais condições de compra são verdadeiras
        if sum([condicao_compra_1, condicao_compra_2, condicao_compra_3, condicao_compra_4, condicao_compra_5, condicao_compra_6, condicao_compra_7, condicao_compra_8, condicao_compra_9, condicao_compra_10, condicao_compra_11, condicao_compra_12, condicao_compra_13, condicao_compra_14, condicao_compra_15, condicao_compra_16, condicao_compra_17, condicao_compra_18, condicao_compra_19, condicao_compra_20]) >= 8:
            print("Tomando decisão de compra para {pair}")
            comprar('EURUSD', 50, 1)  # Compra EURUSD com valor de $1 e tempo de 1 minuto
    
         # Verifica se duas ou mais condições de venda são verdadeiras
        elif sum([condicao_venda_1, condicao_venda_2, condicao_venda_3, condicao_venda_4, condicao_venda_5, condicao_venda_6, condicao_venda_7, condicao_venda_8, condicao_venda_9, condicao_venda_10, condicao_venda_11, condicao_venda_12, condicao_venda_13, condicao_venda_14, condicao_venda_15, condicao_venda_16, condicao_venda_17, condicao_venda_18, condicao_venda_19, condicao_venda_20]) >= 8:
            print("Tomando decisão de venda para {pair}")
            vender('EURUSD', 50, 1)  # Venda EURUSD com valor de $1 e tempo de 1 minuto
        
          # Adicione um atraso entre os trades para evitar a execução demasiado rápida de ordens
        time.sleep(60)
            # Adicionando os casos de erro ao conjunto de dados de treinamento
        X_train = np.concatenate((X_train, X_test[indices_de_erros]))
        y_train = np.concatenate((y_train, y_test.iloc[indices_de_erros]))  
         
           # Salvando o DataFrame df em um arquivo CSV para futuros treinamentos
    df.to_csv('analises_e_entradas.csv', index=False)
    print("Dados salvos com sucesso no arquivo 'analises_e_entradas.csv'")
    
   # Salvando o modelo atualizado
    model_1.save("meu_modelo_1.keras")
  
    print(f"Modelos atualizados e salvo no dia")
        
    time.sleep(60)