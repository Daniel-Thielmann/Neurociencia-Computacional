import numpy as np
import matplotlib.pyplot as plt
import imageio
from io import BytesIO
import json
import os

# Função para carregar parâmetros de um arquivo JSON


def parametros_json(json_file):
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            params = json.load(f)
    else:
        raise FileNotFoundError(f"Arquivo '{json_file}' não encontrado.")
    return params

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

    # Inicializar variáveis
    n_x = int(L_max / dx) + 1
    n_t = int(T_max / dt) + 1
    V = np.ones(n_x) * V_m0
    n = np.ones(n_x) * n0
    m = np.ones(n_x) * m0
    h = np.ones(n_x) * h0
    V_time = np.zeros((n_t, n_x))

    # Constante difusiva D
    D = (a / (2 * R_l)) * (dt / dx**2)

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
            dV_dt = (D * (V[x_idx+1] - 2 * V[x_idx] + V[x_idx-1]
                          ) - (I_Na + I_K + I_L - I_stim) * dt) / C_m
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

        # Atualizar V e armazenar o tempo
        V = V_new
        V_time[t_idx, :] = V

    return V_time

# Função para criar o GIF


def create_hodgkin_huxley_gif(V_time, dx, L_max, dt, y_amplitude=(-100, 100), frame_skip=1):
    n_t, _ = V_time.shape
    x = np.linspace(0, L_max, V_time.shape[1])
    y_min, y_max = y_amplitude
    frames = []

    for t_idx in range(0, n_t, frame_skip):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, V_time[t_idx, :], label=f"t = {
                t_idx * dt:.1f} ms", color="blue")
        ax.set_xlabel("Posição x (cm)")
        ax.set_ylabel("Potencial de membrana V (mV)")
        ax.set_title(
            "Propagação do Potencial de Ação - Modelo de Hodgkin-Huxley 1D")
        ax.set_ylim(y_min, y_max)
        ax.legend()
        ax.grid()

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        frames.append(imageio.v2.imread(buf))
        buf.close()
        plt.close(fig)

    gif_bytes = BytesIO()
    with imageio.get_writer(gif_bytes, mode="I", duration=dt * frame_skip / 1000, format='GIF') as writer:
        for frame in frames:
            writer.append_data(frame)

    gif_bytes.seek(0)
    return gif_bytes


# Executar a simulação
params = parametros_json("parametros.json")
V_time = hodgkin_huxley_1D(params)

# Criar e salvar o GIF
gif_data = create_hodgkin_huxley_gif(
    V_time, params["dx"], params["L_max"], params["dt"], y_amplitude=(-100, 100), frame_skip=10)
with open("propagacao_potencial1.gif", "wb") as f:
    f.write(gif_data.read())

print("GIF salvo como 'propagacao_potencial1.gif'.")
