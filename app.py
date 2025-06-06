import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(
    page_title="Log-Normal Percentiles",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("📊 Distribución Log-Normal a partir de dos percentiles")
st.write("""
Esta aplicación calcula un tercer percentil de una distribución log-normal 
cuando tú proporcionas dos valores correspondientes a dos percentiles fijos:
- **Típico** (50 %)
- **Serio**  (80 %)
- **Extremo** (95 %)

Luego de ingresar dos de esos valores, se ajusta μ y σ del modelo log-normal,
y se obtiene el valor faltante para el tercer percentil. Además, mostramos varias gráficas:
1. PDF de la distribución con sus tres percentiles.
2. Boxplot de los valores.
3. Bar chart comparativo.
4. Dot‐plot sobre un único eje (solo eje Y).
""")

# Diccionario percentiles
percentiles = {
    "Típico (50 %)": 0.50,
    "Serio  (80 %)": 0.80,
    "Extremo (95 %)": 0.95
}

# Sidebar con instrucciones
with st.sidebar:
    st.header("¿Cómo usar?")
    st.write("""
    1. Selecciona exactamente dos percentiles.
    2. Ingresa sus valores (> 0).
    3. La app ajusta la log-normal y calcula el tercero.
    4. Revisa las gráficas generadas.
    """)

# 1) Selección de percentiles
seleccionados = st.multiselect(
    "Selecciona **dos** percentiles:", 
    options=list(percentiles.keys())
)
if len(seleccionados) != 2:
    st.warning("Por favor, selecciona **exactamente dos** percentiles.")
    st.stop()

# 2) Ingreso de valores
st.subheader("Ingresa los valores para los percentiles seleccionados")
valores = {}
for etq in seleccionados:
    valores[etq] = st.number_input(f"Valor para {etq}:", min_value=0.0, format="%.5f")
if any(v is None or v <= 0 for v in valores.values()):
    st.error("Todos los valores deben ser mayores que cero.")
    st.stop()

# 3) Cálculo de μ y σ
etq1, etq2 = seleccionados
x1, x2 = valores[etq1], valores[etq2]
p1, p2 = percentiles[etq1], percentiles[etq2]
z1, z2 = stats.norm.ppf(p1), stats.norm.ppf(p2)
sigma = (np.log(x2) - np.log(x1)) / (z2 - z1)
mu = np.log(x1) - sigma * z1

# 4) Calcular el percentil faltante
etq_falt = [e for e in percentiles if e not in seleccionados][0]
p_falt = percentiles[etq_falt]
z_falt = stats.norm.ppf(p_falt)
x_falt = np.exp(mu + sigma * z_falt)

# Mostrar resultados
st.subheader("Resultados del ajuste")
st.markdown(f"- **μ** = {mu:.5f}    **σ** = {sigma:.5f}")
st.markdown(f"- **{etq_falt}**: {x_falt:.5f}")

# 5) Explicación paso a paso
with st.expander("Ver explicación paso a paso"):
    st.markdown(r"""
    1. Si \(X\) ~ LogNormal(μ, σ²), entonces \(\ln(X)\) ~ Normal(μ, σ²).
    2. Para un percentil \(p\), el valor \(x_p\) satisface:
       \[
         \ln(x_p) = \mu + \sigma\,z_p,
         \quad z_p = \Phi^{-1}(p).
       \]
    3. Dados dos percentiles \((p_1, x_1)\) y \((p_2, x_2)\):
       \[
         \begin{cases}
           \ln(x_1) = \mu + \sigma\,z_1 \\
           \ln(x_2) = \mu + \sigma\,z_2
         \end{cases}
         \;\Longrightarrow\;
         \sigma = \frac{\ln(x_2)-\ln(x_1)}{z_2 - z_1},\quad
         \mu = \ln(x_1) - \sigma\,z_1.
       \]
    4. Con μ y σ, para el percentil faltante \(p_f\) y su \(z_f\):
       \[
         x_f = \exp(\mu + \sigma\,z_f).
       \]
    5. Así obtenemos el valor del percentil no proporcionado.
    """)

# Preparamos listas de etiquetas y valores
etq_all = [etq1, etq2, etq_falt]
x_all = [x1, x2, x_falt]
y_perc = [percentiles[e]*100 for e in etq_all]

# 1) PDF con percentiles
st.subheader("1. PDF con los tres percentiles")
xs = np.linspace(min(x_all)*0.5, max(x_all)*1.5, 500)
pdf = stats.lognorm.pdf(xs, s=sigma, scale=np.exp(mu))
fig1, ax1 = plt.subplots(figsize=(6, 4))
ax1.plot(xs, pdf, lw=2, label="PDF Log-Normal")
for etq, xv in zip(etq_all, x_all):
    ax1.axvline(xv, ls="--", lw=2)
    ax1.text(xv, max(pdf)*0.06, f"{etq.split()[0]}\n{xv:.2f}",
             rotation=90, va="bottom", ha="center")
ax1.set_xlabel("Valor")
ax1.set_ylabel("Densidad")
ax1.legend(fontsize=8)
st.pyplot(fig1)

# 2) Boxplot
st.subheader("2. Boxplot de los valores")
fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.boxplot([[x] for x in x_all], labels=etq_all)
ax2.set_ylabel("Valor")
ax2.set_title("Boxplot de valores percentiles")
st.pyplot(fig2)

# 3) Bar chart comparativo
st.subheader("3. Bar chart comparativo")
fig3, ax3 = plt.subplots(figsize=(6, 4))
ax3.bar(etq_all, x_all)
ax3.set_ylabel("Valor")
ax3.set_title("Comparación de valores percentiles")
st.pyplot(fig3)

# 4) Dot‐plot sobre un único eje (solo eje Y)
st.subheader("4. Dot‐plot sobre un único eje (solo eje Y)")
fig4, ax4 = plt.subplots(figsize=(2, 4))
# todos los puntos en x = 0
ax4.scatter([0, 0, 0], y_perc, s=100)
# ocultar eje X
ax4.get_xaxis().set_visible(False)
# ocultar spines innecesarios
for spine in ["top", "right", "bottom"]:
    ax4.spines[spine].set_visible(False)
# Anotar valores a la derecha de los puntos
for yv, xv in zip(y_perc, x_all):
    ax4.text(0.1, yv, f"{xv:.2f}", va="center", ha="left")
ax4.set_ylabel("Percentil (%)")
ax4.set_yticks(sorted(y_perc))
ax4.set_ylim(0, 100)
ax4.set_title("Valores sobre el eje Y")
st.pyplot(fig4)
