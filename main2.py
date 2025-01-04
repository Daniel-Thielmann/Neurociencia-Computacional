import numpy as np
import matplotlib.pyplot as plt
import imageio
from io import BytesIO

# Parâmetros da equação do cabo e do modelo Hodgkin-Huxley
C_m = 1.0        # Capacitância da membrana (µF/cm²)
g_Na = 120.0     # Condutância máxima de Na+ (mS/cm²)
g_K = 36.0       # Condutância máxima de K+ (mS/cm²)
g_L = 0.3        # Condutância de vazamento (mS/cm²)
E_Na = 50.0      # Potencial de reversão de Na+ (mV)
E_K = -77.0      # Potencial de reversão de K+ (mV)
E_L = -54.387    # Potencial de reversão de vazamento (mV)

a = 15       # raio do axônio (µm)
R = 5000        # Resistência longitudinal (ohm·cm)

# Parâmetros do gráfico e do GIF
dx = 0.01        # Espaçamento espacial (cm)
dt = 0.01        # Passo de tempo (ms)
L = 3.0          # Comprimento total da fibra (cm)
T = 50.0         # Tempo total de simulação (ms)

# Função para corrente de estímulo aplicada


def I_ap(t, x):
    if x == dx:
        return 20.0  # µA/cm²
    return 0.0


def safe_exp(x):
    # Trunca valores extremos para evitar overflow
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


# Simulação do modelo Hodgkin-Huxley com equação do cabo
def hodgkin_huxley_1D():
    n_x = int(L / dx)  # Número de pontos espaciais
    n_t = int(T / dt)  # Número de passos temporais

    # Variáveis de estado
    V = np.ones(n_x) * -65.0  # Potencial inicial (mV)
    n = np.zeros(n_x) + 0.3177
    m = np.zeros(n_x) + 0.0529
    h = np.zeros(n_x) + 0.5961

    # Armazenamento para visualização
    V_time = np.zeros((n_t, n_x))

    # Constante difusiva D
    D = (a / (2 * R)) * (dt / dx**2)

    # Iteração temporal
    for t_idx in range(n_t):

        if t_idx % 100 == 0:
            print(f"Tempo: {t_idx * dt:.2f} ms")

        V_new = V.copy()

        V_new[0] = V_new[1]
        V_new[0] = V_new[1]

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
            I_stim = I_ap(t_idx * dt, x_idx * dx)

            # Atualização do potencial usando a equação do cabo
            dV_dt = (D * (V[x_idx+1] - 2 * V[x_idx] + V[x_idx-1]
                          ) - (I_Na + I_K + I_L - I_stim)*dt) / C_m
            V_new[x_idx] = V[x_idx] + dV_dt
            # V_new[x_idx] = np.clip(V_new[x_idx], -100, 100)  # Valores realistas para o potencial
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


def hodgkin_huxley_1D_with_myelin():
    n_x = int(L / dx)  # Número de pontos espaciais
    n_t = int(T / dt)  # Número de passos temporais

    # Variáveis de estado
    V = np.ones(n_x) * -65.0  # Potencial inicial (mV)
    n = np.zeros(n_x) + 0.3177
    m = np.zeros(n_x) + 0.0529
    h = np.zeros(n_x) + 0.5961

    # Vetor de mielina (1 para mielina presente, 0 para ausente)
    mielina = np.zeros(n_x)
    # Mielina a cada 0.2 cm (ajuste conforme necessário)
    mielina[::int(0.2 / dx)] = 1

    # Modificar os parâmetros com base na mielina
    # Capacitância reduzida onde há mielina
    C_m_mod = C_m * (1 - mielina) + (C_m * 0.1) * mielina
    # Condutância reduzida onde há mielina
    g_L_mod = g_L * (1 - mielina) + (g_L * 0.1) * mielina

    # Armazenamento para visualização
    V_time = np.zeros((n_t, n_x))

    # Iteração temporal
    for t_idx in range(n_t):
        V_new = V.copy()

        # Condições de contorno (isolamento nos extremos)
        V_new[0] = V_new[1]
        V_new[-1] = V_new[-2]

        for x_idx in range(1, n_x - 1):
            # Correntes iônicas
            I_Na = g_Na * m[x_idx]**3 * h[x_idx] * (V[x_idx] - E_Na)
            I_K = g_K * n[x_idx]**4 * (V[x_idx] - E_K)
            I_L = g_L_mod[x_idx] * (V[x_idx] - E_L)
            I_stim = I_ap(t_idx * dt, x_idx * dx)

            # Atualização do potencial usando a equação do cabo
            D = (a / (2 * R)) * (dt / dx**2) * \
                (1 - mielina[x_idx] + mielina[x_idx] * 10)
            dV_dt = D * (V[x_idx+1] - 2 * V[x_idx] + V[x_idx-1]) - \
                (I_Na + I_K + I_L - I_stim) / C_m_mod[x_idx]
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


def create_hodgkin_huxley_gif(V_time, dx, dt, y_amplitude=(-100, 100), frame_skip=1):
    n_t, n_x = V_time.shape
    x = np.linspace(0, L, n_x)  # Espaço
    y_min, y_max = y_amplitude  # Amplitude fixa
    frames = []  # Lista para armazenar as imagens em memória

    for t_idx in range(0, n_t, frame_skip):
        # Configura o gráfico
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, V_time[t_idx, :], label=f"t = {
                t_idx * dt:.1f} ms", color="blue")
        ax.set_xlabel("Posição x (cm)")
        ax.set_ylabel("Potencial de membrana V (mV)")
        ax.set_title(
            "Propagação do Potencial de Ação - Modelo de Hodgkin-Huxley 1D")
        ax.set_ylim(y_min, y_max)  # Mantém a escala do gráfico fixa
        ax.legend()
        ax.grid()

        # Salvar o frame em memória
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        frames.append(imageio.v2.imread(buf))
        buf.close()
        plt.close(fig)

    # Criar o GIF diretamente
    gif_bytes = BytesIO()
    with imageio.get_writer(gif_bytes, mode="I", duration=dt * frame_skip / 1000, format='GIF') as writer:
        for frame in frames:
            writer.append_data(frame)

    gif_bytes.seek(0)
    return gif_bytes


# Executar a simulação
V_time = hodgkin_huxley_1D()

# Criar o GIF em memória
gif_data = create_hodgkin_huxley_gif(
    V_time, dx, dt, y_amplitude=(-100, 100), frame_skip=10)

# Salvar o GIF em um arquivo ou usar diretamente
with open("propagacao_potencial.gif", "wb") as f:
    f.write(gif_data.read())

print("GIF salvo como 'propagacao_potencial.gif'.")
