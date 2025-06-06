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

Luego de ingresar dos de esos valores, se ajusta \(\mu\) y \(\sigma\) del modelo log-normal,
y se obtiene el valor faltante para el tercer percentil. Además, mostramos la curva de densidad 
y señalamos los tres puntos sobre ella.
""")

# Diccionario con mapeo etiqueta → probabilidad
percentiles = {
    "Típico (50 %)": 0.50,
    "Serio  (80 %)": 0.80,
    "Extremo (95 %)": 0.95
}

# Sidebar para explicación breve
with st.sidebar:
    st.header("¿Cómo usar?")
    st.write("""
    1. En **Seleccionar percentiles**, elige exactamente dos de los tres.
    2. Ingresa los **valores** correspondientes a esos dos.
    3. La aplicación ajusta automáticamente la distribución log-normal 
       y calcula el valor faltante.
    4. Verás:
       - El valor del percentil calculado.
       - La gráfica de densidad (PDF) de la log-normal.
       - Una explicación paso a paso.
    """)

# 1) Selección de dos percentiles
seleccionados = st.multiselect(
    "Selecciona **dos** percentiles para ingresar sus valores:", 
    options=list(percentiles.keys()),
    help="Elige exactamente dos de los tres: Típico, Serio o Extremo."
)

# Validación: deben seleccionarse exactamente dos
if len(seleccionados) != 2:
    st.warning("Por favor, selecciona **exactamente dos** percentiles para continuar.")
    st.stop()

# 2) Pedir los valores numéricos para los percentiles escogidos
st.subheader("Ingresa los valores para cada percentil seleccionado")
valores = {}
for etiqueta in seleccionados:
    valores[etiqueta] = st.number_input(
        f"Valor para «{etiqueta}»:", 
        min_value=0.0, 
        format="%.5f"
    )

# Una vez ingresados, procedemos al cálculo
if all(val is not None for val in valores.values()):
    # Obtener (p1, x1) y (p2, x2)
    etq1, etq2 = seleccionados
    p1, p2 = percentiles[etq1], percentiles[etq2]
    x1, x2 = valores[etq1], valores[etq2]

    if x1 <= 0 or x2 <= 0:
        st.error("Los valores deben ser mayores que cero para una distribución log-normal.")
        st.stop()

    # 3) Cálculo de mu y sigma de la normal subyacente:
    #    ln(x1) = mu + sigma * z1,  ln(x2) = mu + sigma * z2
    z1 = stats.norm.ppf(p1)
    z2 = stats.norm.ppf(p2)
    sigma = (np.log(x2) - np.log(x1)) / (z2 - z1)
    mu = np.log(x1) - sigma * z1

    # 4) Determinar cuál etiqueta falta
    etiqueta_faltante = [e for e in percentiles.keys() if e not in seleccionados][0]
    p_falt = percentiles[etiqueta_faltante]
    z_falt = stats.norm.ppf(p_falt)
    x_falt = np.exp(mu + sigma * z_falt)

    # Mostrar resultados
    st.subheader("Resultados del ajuste")
    st.markdown(f"- **Parámetros estimados** de la normal subyacente (logarítmica):\n"
                f"  - μ = {mu:.5f}\n"
                f"  - σ = {sigma:.5f}")
    st.markdown(f"- **Percentil faltante**: «{etiqueta_faltante}» ({p_falt*100:.0f} %)\n"
                f"  - Valor calculado: **{x_falt:.5f}**")

    # 5) Explicación detallada paso a paso
    with st.expander("Ver explicación paso a paso"):
        st.markdown("""
        1. Partimos de que si \(X\) ~ LogNormal(μ, σ²), entonces \( \ln(X)\) ~ Normal(μ, σ²).
        2. Para un percentil \(p\), el valor \(x_p\) satisface:
           \[
           \ln(x_p) = \mu \;+\; \sigma \cdot z_p,
           \quad\text{donde}\;z_p = Φ^{-1}(p)\;(percentil\;de\;N(0,1)).
           \]
        3. Dado \((p_1,\,x_1)\) y \((p_2,\,x_2)\), resolvemos el sistema:
           \[
           \begin{cases}
             \ln(x_1) = \mu + \sigma z_1 \\
             \ln(x_2) = \mu + \sigma z_2
           \end{cases}
           \;\Longrightarrow\;
           \begin{aligned}
             \sigma &= \frac{\ln(x_2) - \ln(x_1)}{z_2 - z_1}, \\
             \mu    &= \ln(x_1) - \sigma\,z_1.
           \end{aligned}
           \]
        4. Una vez μ y σ conocidos, para el percentil faltante \(p_f\) (con su z-score \(z_f\)):
           \[
           \ln(x_f) = \mu + \sigma\,z_f
           \quad\Longrightarrow\quad
           x_f = \exp\bigl(\mu + \sigma\,z_f\bigr).
           \]
        5. Así obtenemos el valor del percentil que no proporcionaste.
        """)

    # 6) Graficar la PDF de la log-normal y marcar los tres puntos
    st.subheader("Gráfica de densidad (PDF) con los tres percentiles")
    # Elegir rango de x para graficar: desde un % muy bajo hasta algo mayor que el máximo ingresado/calculado
    todos_x = [x1, x2, x_falt]
    xmin = max(min(todos_x) * 0.5, 1e-3)
    xmax = max(todos_x) * 1.5
    xs = np.linspace(xmin, xmax, 500)
    pdf = stats.lognorm.pdf(xs, s=sigma, scale=np.exp(mu))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, pdf, lw=2, label="PDF Log-Normal")
    # Marcar verticales en los tres percentiles
    colores = {"Típico (50 %)": "tab:blue", 
               "Serio  (80 %)": "tab:orange", 
               "Extremo (95 %)": "tab:green"}
    for etq, prob in percentiles.items():
        # obtener x: si es uno de los dos ingresados, usar x1/x2; si es el faltante, usar x_falt
        if etq in valores:
            x_val = valores[etq]
        else:
            x_val = x_falt
        ax.axvline(x_val, color=colores[etq], ls="--", lw=2)
        ax.text(x_val, 
                max(pdf)*0.06, 
                f"{etq.split()[0]}\n{x_val:.2f}", 
                rotation=90, 
                va="bottom", 
                ha="center", 
                color=colores[etq],
                fontsize=9)
    ax.set_xlabel("x")
    ax.set_ylabel("Densidad")
    ax.set_title("Distribución Log-Normal ajustada")
    ax.legend(loc="upper right", fontsize=8)
    st.pyplot(fig)

    # 7) Mostrar con más detalle los valores finales
    st.subheader("Resumen de los tres percentiles")
    st.markdown(f"""
    | Percentil      | Probabilidad | Valor         |
    | -------------- |:------------:| -------------:|
    | Típico         | 50 %         | { (valores.get('Típico (50 %)', x_falt if etiqueta_faltante=='Típico (50 %)' else None)) :.5f} |
    | Serio          | 80 %         | { (valores.get('Serio  (80 %)', x_falt if etiqueta_faltante=='Serio  (80 %)' else None)) :.5f} |
    | Extremo        | 95 %         | { (valores.get('Extremo (95 %)', x_falt if etiqueta_faltante=='Extremo (95 %)' else None)) :.5f} |
    """)

    st.success("¡Listo! Si cambias cualquiera de los dos valores ingresados, la distribución y el percentil calculado se actualizarán automáticamente.")
