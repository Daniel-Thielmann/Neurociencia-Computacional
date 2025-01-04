import numpy as np
import matplotlib.pyplot as plt
import imageio
from io import BytesIO
import json
import os
import sys
import csv

# Função para carregar parâmetros de um arquivo JSON


def parametros_json(json_file):
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            params = json.load(f)
    else:
        raise FileNotFoundError(f"Arquivo '{json_file}' não encontrado.")
    return params

# Função para carregar o arquivo config.txt


def le_config_txt(txt_file):
    config = {}
    with open(txt_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue  # Ignorar comentários e linhas vazias
            key, value = map(str.strip, line.split("=", 1))
            config[key] = value
    return config

# Funções auxiliares para os canais


def safe_exp(x):
    # Limitar o valor de x para evitar overflow
    return np.exp(np.clip(x, -100, 100))


def alpha_n(V):
    return 0.01 * (V + 55) / (1 - safe_exp(-(V + 55) / 10))


def beta_n(V):
    return 0.125 * safe_exp(-(V + 65) / 80)


def alpha_m(V):
    return 0.1 * (V + 40) / (1 - safe_exp(-(V + 40) / 10))


def beta_m(V):
    return 4.0 * safe_exp(-(V + 65) / 18)


def alpha_h(V):
    return 0.07 * safe_exp(-(V + 65) / 20)


def beta_h(V):
    return 1 / (1 + safe_exp(-(V + 35) / 10))

# Função de simulação Hodgkin-Huxley 1D


def hodgkin_huxley_1D(params):
    # Extrair parâmetros
    C_m = params["cm"]
    a = params["a"]
    R_l = params["rl"]
    g_Na = params["gna"]
    g_K = params["gk"]
    g_L = params["gl"]
    E_Na = params["ena"]
    E_K = params["ek"]
    E_L = params["el"]
    T_max = params["T_max"]
    L_max = params["L_max"]
    dt = params["dt"]
    dx = params["dx"]
    V_m0 = params["vm0"]
    m0 = params["m0"]
    h0 = params["h0"]
    n0 = params["n0"]
    J = np.array(params["J"])
    Mie = np.array(params.get("Mie", np.zeros(int(L_max / dx) + 1)))

    # Inicializar variáveis
    n_x = int(L_max / dx) + 1
    n_t = int(T_max / dt) + 1
    V = np.ones(n_x) * V_m0
    n = np.ones(n_x) * n0
    m = np.ones(n_x) * m0
    h = np.ones(n_x) * h0
    V_time = np.zeros((n_t, n_x))

    # Constante difusiva D
    D_base = (a / (2 * R_l)) * (dt / dx**2)
    D = np.where(Mie == 1, D_base * 5, D_base)

    # Iteração temporal
    for t_idx in range(n_t):
        if t_idx % 100 == 0:
            print(f"Tempo: {t_idx * dt:.2f} ms")

        V_new = V.copy()
        V_new[0] = V_new[1]
        V_new[-1] = V_new[-2]

        n[0] = n[1]
        m[0] = m[1]
        h[0] = h[1]

        n[-1] = n[-2]
        m[-1] = m[-2]
        h[-1] = h[-2]

        for x_idx in range(1, n_x - 1):
            # Correntes iônicas
            I_Na = g_Na * m[x_idx]**3 * h[x_idx] * (V[x_idx] - E_Na)
            I_K = g_K * n[x_idx]**4 * (V[x_idx] - E_K)
            I_L = g_L * (V[x_idx] - E_L)
            I_stim = J[t_idx, x_idx]

            # Atualização do potencial usando a equação do cabo
            dV_dt = (D[x_idx] * (V[x_idx+1] - 2 * V[x_idx] +
                     V[x_idx-1]) - (I_Na + I_K + I_L - I_stim) * dt) / C_m
            V_new[x_idx] = V[x_idx] + dV_dt

            # Atualização dos gates
            dn = (alpha_n(V[x_idx]) * (1 - n[x_idx]) -
                  beta_n(V[x_idx]) * n[x_idx]) * dt
            dm = (alpha_m(V[x_idx]) * (1 - m[x_idx]) -
                  beta_m(V[x_idx]) * m[x_idx]) * dt
            dh = (alpha_h(V[x_idx]) * (1 - h[x_idx]) -
                  beta_h(V[x_idx]) * h[x_idx]) * dt

            n[x_idx] += dn
            m[x_idx] += dm
            h[x_idx] += dh

        # Handle overflow and NaN values
        V_new = np.nan_to_num(V_new, nan=0.0, posinf=100.0, neginf=-100.0)
        n = np.nan_to_num(n, nan=0.0, posinf=1.0, neginf=0.0)
        m = np.nan_to_num(m, nan=0.0, posinf=1.0, neginf=0.0)
        h = np.nan_to_num(h, nan=0.0, posinf=1.0, neginf=0.0)

        # Atualizar V e armazenar o tempo
        V = V_new
        V_time[t_idx, :] = V

    return V_time, n, m, h

# Função para criar gráficos comparativos


def save_comparison_plot(V_time, Mie, dx, L_max, filename="comparison.png"):
    x = np.linspace(0, L_max, V_time.shape[1])
    plt.figure(figsize=(12, 6))
    plt.plot(x, V_time[-1, :], label="Com Mielina")
    plt.plot(x, V_time[0, :], label="Sem Mielina")
    plt.title("Comparação do Potencial de Membrana com e sem Bainha de Mielina")
    plt.xlabel("Posição (cm)")
    plt.ylabel("Potencial de Membrana (mV)")
    plt.legend()
    plt.grid()
    # Marcar região com mielina em vermelho
    for i in range(len(Mie)):
        if Mie[i] == 1:
            plt.axvspan(i * dx, (i + 1) * dx, color='red', alpha=0.3)
    plt.savefig(filename)
    plt.close()

# Função para gráficos dos canais iônicos


def save_ion_channel_plot(n, m, h, L_max, dx, filename="ion_channels.png"):
    x = np.linspace(0, L_max, len(n))
    plt.figure(figsize=(12, 6))
    plt.plot(x, n, label="Canal n", color="blue", linewidth=2)
    plt.plot(x, m, label="Canal m", color="orange", linewidth=2)
    plt.plot(x, h, label="Canal h", color="green", linewidth=2)
    plt.title("Abertura dos Canais Iônicos")
    plt.xlabel("Posição (cm)")
    plt.ylabel("Probabilidade de Abertura")
    plt.ylim([0, 1])  # Ajusta o intervalo para probabilidades
    plt.xlim([0, L_max])
    plt.legend()
    plt.grid()
    plt.savefig(filename)
    plt.close()

# Função para criar o GIF


def create_hodgkin_huxley_gif(V_time, dx, L_max, dt, Mie, y_amplitude=(-100, 100), frame_skip=1, filename="propagacao_potencial1.gif"):
    n_t, _ = V_time.shape
    x = np.linspace(0, L_max, V_time.shape[1])
    y_min, y_max = y_amplitude

    print(f"Iniciando geração de frames para o GIF...")

    with imageio.get_writer(filename, mode="I", duration=dt * frame_skip / 1000) as writer:
        for t_idx in range(0, n_t, frame_skip):
            # print(f"Criando frame {t_idx}/{n_t}...")
            plt.figure(figsize=(10, 6))
            plt.plot(x, V_time[t_idx, :], label=f"t = {t_idx * dt:.1f} ms", color="blue")
            plt.xlabel("Posição x (cm)")
            plt.ylabel("Potencial de membrana V (mV)")
            plt.title("Propagação do Potencial de Ação - Modelo de Hodgkin-Huxley 1D")
            plt.ylim(y_min, y_max)
            plt.legend()
            plt.grid()
            # Marcar região com mielina em vermelho
            for i in range(len(Mie)):
                if Mie[i] == 1:
                    plt.axvspan(i * dx, (i + 1) * dx, color='red', alpha=0.3)

            buf = BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            writer.append_data(imageio.v2.imread(buf))
            buf.close()
            plt.close()

    print(f"GIF gerado com sucesso: {filename}")

def table_csv(V_time, dt, n, m, h, filename='output.csv'):
    print(f"Salvando dados em {filename}...")
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time', 'V_time', 'n', 'm', 'h'])
        for i, (V, n_val, m_val, h_val) in enumerate(zip(V_time, n, m, h), start=1):
            V_str = ','.join(map(str, V))
            n_str = str(n_val)
            m_str = str(m_val)
            h_str = str(h_val)
            writer.writerow([i*dt, V_str, n_str, m_str, h_str])


# Verificar argumentos da linha de comando
if len(sys.argv) < 2:
    print("Uso: python main.py config.txt")
    sys.exit(1)

# Carregar configurações do config.txt
config_file = sys.argv[1]
config = le_config_txt(config_file)
print(f"Descrição: {config.get('descricao', 'Não especificada')}")
print(f"Autor: {config.get('autor', 'Não especificado')}")
print(f"Data: {config.get('data', 'Não especificada')}")
print(f"Versão: {config.get('versao', 'Não especificada')}")

# Carregar parâmetros do parametros.json
params = parametros_json("parametros.json")

# Simulação
V_time, n_final, m_final, h_final = hodgkin_huxley_1D(params)

# Gráficos
save_comparison_plot(V_time, params.get("Mie", np.zeros(int(
    params["L_max"] / params["dx"]) + 1)), params["dx"], params["L_max"], filename="comparison.png")
save_ion_channel_plot(n_final, m_final, h_final,
                      params["L_max"], params["dx"], filename="ion_channels.png")
create_hodgkin_huxley_gif(
    V_time, params["dx"], params["L_max"], params["dt"], params["Mie"], y_amplitude=(-100, 100), frame_skip=10, filename="propagacao_potencial1.gif")

# Salvar dados em um arquivo CSV
table_csv(V_time, params["dt"], n_final, m_final, h_final, filename='output.csv')
