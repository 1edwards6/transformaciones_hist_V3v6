# -*- coding: utf-8 -*-
"""Streamlit — V3v6: modelo de histograma con cuadrícula jerárquica por gradiente."""
from __future__ import annotations
import io
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Rectangle
from PIL import Image

from modelo_v3 import procesar_imagen_v3v6

st.set_page_config(page_title="V3v6 · Cuadrícula jerárquica por gradiente", page_icon="🖼️", layout="wide")
st.title("V3v6 · Mejoramiento matemático de imágenes")
st.markdown(
    "La **V3** es el núcleo matemático: expande y reacomoda el histograma para construir una nueva imagen. "
    "La **V3v2** incorpora mapas de transformación locales. La **V3v6 experimental** añade dos niveles de "
    "selección adaptable por gradiente: primero divide la imagen en cuatro cuadrantes y luego vuelve a buscar "
    "una partición local dentro de cada cuadrante, obteniendo hasta **16 regiones finales**. "
    "Las transformaciones regionales se interpolan espacialmente para reducir fronteras bruscas."
)

with st.sidebar:
    st.header("Imagen")
    archivo = st.file_uploader("Carga una imagen", type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"])
    st.divider()
    st.header("Modelo V3: histograma")
    porcentaje = st.slider("Porcentaje cubierto por el intervalo mínimo", 80.0, 100.0, 97.5, 0.5,
                           help="La V3 busca el intervalo de intensidades más corto que cubre el porcentaje seleccionado.")
    metodo = st.selectbox("Aproximación del histograma", ["PCHIP", "LINEAL", "FOURIER"], index=0)
    puntos = st.slider("Puntos de aproximación (PCHIP/LINEAL)", 2, 256, 150, 1)
    armonicos = st.slider("Número de armónicos (FOURIER)", 0, 64, 12, 1)
    st.caption("Los controles de puntos y armónicos solo afectan al método correspondiente.")
    st.divider()
    st.header("Cuadrícula jerárquica por gradiente")
    paso = st.select_slider("Paso de búsqueda (%)", options=[2.5, 5.0, 7.5, 10.0], value=5.0)
    banda = st.slider("Banda exterior mínima candidata (%)", 5.0, 35.0, 10.0, 2.5,
                      help="Se ajusta si contradice el tamaño mínimo solicitado para la región central candidata.")
    region_min = st.slider("Ancho/alto mínimo de región central (%)", 15.0, 70.0, 30.0, 5.0)
    peso_ext = st.slider("Peso de penalización del gradiente exterior", 0.0, 1.5, 0.55, 0.05,
                         help="Se maximiza J = gradiente central − peso × gradiente medio exterior.")
    suavizado = st.slider("Suavizado previo σ", 0.0, 5.0, 2.0, 0.25,
                          help="Un valor mayor reduce el ruido fino antes de calcular derivadas.")
    st.caption("El proceso se repite una sola vez dentro de los cuatro cuadrantes: hasta 16 regiones finales.")
    st.divider()
    st.header("Regiones locales")
    fusionar = st.checkbox("Fusionar regiones con baja dispersión", value=True)
    umbral = st.slider("Umbral de desviación estándar para fusionar", 0.0, 60.0, 10.0, 0.5,
                       disabled=not fusionar)
    proteger = st.checkbox("Proteger regiones constantes", value=True,
                           help="Las regiones con un único nivel de intensidad conservan la transformación identidad.")
    st.divider()
    st.header("Método de referencia CLAHE")
    clahe_limite = st.slider("Límite de contraste CLAHE", 0.5, 5.0, 2.0, 0.25)
    clahe_grid = st.slider("Cuadrícula CLAHE", 2, 16, 8, 1)

if archivo is None:
    st.info("Carga una imagen desde el panel lateral para ejecutar la V3v6.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("V3 · Histograma")
        st.write("Calcula el intervalo de intensidades, aproxima su distribución y construye un histograma objetivo con la frecuencia total ajustada al número de píxeles.")
    with c2:
        st.subheader("V3v2 · Regiones locales")
        st.write("Obtiene mapas de transformación locales y los combina espacialmente para reducir transiciones bruscas entre regiones.")
    with c3:
        st.subheader("V3v6 · Cuadrícula jerárquica")
        st.write("Selecciona una primera cuadrícula 2×2 por gradiente; después repite la búsqueda en cada cuadrante y procesa hasta 16 regiones locales.")
    st.stop()

image_bytes = archivo.getvalue()
params = {
    "porcentaje_conservado": porcentaje,
    "metodo": metodo,
    "num_puntos": puntos,
    "num_armonicos": armonicos,
    "paso_busqueda": paso,
    "banda_minima": banda,
    "region_minima": region_min,
    "peso_exterior": peso_ext,
    "suavizado_gradiente": suavizado,
    "fusionar_celdas": fusionar,
    "umbral_desviacion": umbral,
    "proteger_constantes": proteger,
    "clahe_limite": clahe_limite,
    "clahe_grid": clahe_grid,
}

@st.cache_data(show_spinner=False, max_entries=3)
def _procesar_cache(datos_imagen: bytes, parametros_serializados: tuple):
    imagen_pil = Image.open(io.BytesIO(datos_imagen)).convert("RGB")
    rgb = np.asarray(imagen_pil, dtype=np.uint8)
    nombres = [
        "porcentaje_conservado", "metodo", "num_puntos", "num_armonicos", "paso_busqueda",
        "banda_minima", "region_minima", "peso_exterior", "suavizado_gradiente",
        "fusionar_celdas", "umbral_desviacion", "proteger_constantes", "clahe_limite", "clahe_grid"
    ]
    return procesar_imagen_v3v6(rgb, dict(zip(nombres, parametros_serializados)))

params_tuple = tuple(params[k] for k in [
    "porcentaje_conservado", "metodo", "num_puntos", "num_armonicos", "paso_busqueda",
    "banda_minima", "region_minima", "peso_exterior", "suavizado_gradiente",
    "fusionar_celdas", "umbral_desviacion", "proteger_constantes", "clahe_limite", "clahe_grid"
])

try:
    with st.spinner("Calculando V3 global, buscando gradiente en dos niveles y generando mapas locales…"):
        R = _procesar_cache(image_bytes, params_tuple)
except Exception as exc:
    st.error(f"No fue posible procesar la imagen: {exc}")
    st.exception(exc)
    st.stop()

D = R["diagnostico_seleccion"]
region_candidata = D["region_candidata_3x3"]

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Gradiente central nivel 1", f"{D['gradiente_medio_central']:.3f}")
m2.metric("Gradiente exterior nivel 1", f"{D['gradiente_medio_exterior']:.3f}")
m3.metric("Puntuación J nivel 1", f"{D['puntaje']:.3f}")
m4.metric("Regiones finales", str(len(R["regiones_finales"])))
m5.metric("Grupos de transformación", str(len(R["grupos"])))
if abs(D["banda_minima_efectiva_pct"] - banda) > 1e-6:
    st.info(f"Para permitir una región central de al menos {region_min:.1f} %, la banda exterior mínima efectiva se ajustó de {banda:.1f} % a {D['banda_minima_efectiva_pct']:.1f} %.")

# Funciones de exportación
def _png_bytes(array):
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(array).astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()

def _csv_bytes(df, index=True):
    return df.to_csv(index=index).encode("utf-8-sig")

def _make_download_bundle():
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("imagen_original.png", _png_bytes(R["resultados_visuales"]["Original"]))
        for name, im in R["resultados_visuales"].items():
            if name == "Original":
                continue
            safe = name.lower().replace(" ", "_").replace("+", "mas").replace("/", "_")
            zf.writestr(f"{safe}.png", _png_bytes(im))
        zf.writestr("metricas_comparacion.csv", _csv_bytes(R["tabla_metricas"]))
        zf.writestr("diagnostico_16_regiones.csv", _csv_bytes(R["tabla_diagnosticos_locales"], index=False))
        zf.writestr("resumen_regiones_finales.csv", _csv_bytes(R["tabla_regiones_finales"], index=False))
        zf.writestr("busqueda_segundo_nivel.csv", _csv_bytes(R["tabla_subselecciones"], index=False))
        zf.writestr("historial_fusiones.csv", _csv_bytes(R["historial_fusiones"], index=False))
        zf.writestr("mapas_transformacion_locales.csv", _csv_bytes(R["tabla_mapas"], index=False))
        zf.writestr("cdfs_por_region.csv", _csv_bytes(R["tabla_cdfs"], index=False))
        zf.writestr("gradientes_cuadricula_principal_3x3.csv", _csv_bytes(D["tabla_gradientes_3x3"], index=False))
    memory.seek(0)
    return memory.getvalue()

tab_comparacion, tab_grid, tab_hist, tab_local, tab_download = st.tabs([
    "Comparación visual", "Cuadrícula jerárquica", "Histogramas y CDF", "Transformación local", "Diagnóstico y descargas"
])

with tab_comparacion:
    st.subheader("Comparación de los cinco resultados")
    nombres = list(R["resultados_visuales"].keys())
    columnas = st.columns(len(nombres))
    for col, nombre in zip(columnas, nombres):
        with col:
            st.image(R["resultados_visuales"][nombre], caption=nombre, use_container_width=True)
    st.caption("Las métricas se calculan sobre la intensidad en escala de grises. Para imágenes en color, la transformación modifica el canal de luminancia L del espacio Lab.")
    st.subheader("V3 global frente a transformación local jerárquica")
    a, b = st.columns(2)
    a.image(R["resultados_visuales"]["V3 global"], caption="V3 global", use_container_width=True)
    b.image(R["resultados_visuales"][R["nombre_local"]], caption=R["nombre_local"], use_container_width=True)
    st.dataframe(R["tabla_metricas"].round(4), use_container_width=True)

with tab_grid:
    st.subheader("Dos niveles de cuadrícula adaptable")
    st.write(
        f"Nivel 1: la región central candidata ocupa x = {region_candidata['x1_pct']:.1f}–{region_candidata['x2_pct']:.1f} % "
        f"del ancho e y = {region_candidata['y1_pct']:.1f}–{region_candidata['y2_pct']:.1f} % del alto. "
        f"Sus puntos medios producen los ejes globales x = {R['corte_x']} px e y = {R['corte_y']} px."
    )
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].imshow(R["imagen"], cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Cuadrícula jerárquica sobre la imagen")
    axes[0].axis("off")
    axes[0].add_patch(Rectangle(
        (region_candidata["x1_px"] - 0.5, region_candidata["y1_px"] - 0.5),
        region_candidata["x2_px"] - region_candidata["x1_px"],
        region_candidata["y2_px"] - region_candidata["y1_px"],
        fill=False, edgecolor="red", linewidth=1.8, linestyle="--"
    ))
    axes[0].axvline(R["corte_x"] - 0.5, color="navy", linewidth=2.2)
    axes[0].axhline(R["corte_y"] - 0.5, color="navy", linewidth=2.2)
    axes[1].imshow(D["magnitud_gradiente"], cmap="inferno")
    axes[1].set_title("Magnitud del gradiente (nivel 1)")
    axes[1].axis("off")
    axes[1].axvline(R["corte_x"] - 0.5, color="cyan", linewidth=1.8)
    axes[1].axhline(R["corte_y"] - 0.5, color="cyan", linewidth=1.8)
    axes[1].add_patch(Rectangle(
        (region_candidata["x1_px"] - 0.5, region_candidata["y1_px"] - 0.5),
        region_candidata["x2_px"] - region_candidata["x1_px"],
        region_candidata["y2_px"] - region_candidata["y1_px"],
        fill=False, edgecolor="white", linewidth=1.3, linestyle="--"
    ))
    alto, ancho = R["imagen"].shape
    for sub in R["subselecciones"]:
        # El modelo guarda los límites como x_fin/y_fin (coordenadas exclusivas).
        # Se aceptan también los nombres anteriores para mantener compatibilidad.
        cx, cy = sub["corte_x_px"] - 0.5, sub["corte_y_px"] - 0.5
        x_inicio = sub.get("x_inicio", 0)
        x_fin = sub.get("x_fin_exclusivo", sub.get("x_fin", ancho))
        y_inicio = sub.get("y_inicio", 0)
        y_fin = sub.get("y_fin_exclusivo", sub.get("y_fin", alto))
        x0, x1 = x_inicio - 0.5, x_fin - 0.5
        y0, y1 = y_inicio - 0.5, y_fin - 0.5
        axes[0].plot([cx, cx], [y0, y1], color="#00d7e8", linewidth=1.3)
        axes[0].plot([x0, x1], [cy, cy], color="#00d7e8", linewidth=1.3)
        axes[1].plot([cx, cx], [y0, y1], color="#7cf6ff", linewidth=1.15)
        axes[1].plot([x0, x1], [cy, cy], color="#7cf6ff", linewidth=1.15)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown("**Criterio de selección en cada nivel**")
    st.latex(r"J=\overline{G}_{\mathrm{central}}-\lambda\,\overline{G}_{\mathrm{exterior}}")
    st.write("Se maximiza J en la imagen completa. Después, se repite la misma búsqueda en cada uno de los cuatro cuadrantes. Las líneas turquesas son los cortes locales; las líneas azul oscuro, los cortes principales.")
    st.subheader("Búsquedas locales dentro de los cuatro cuadrantes")
    st.dataframe(R["tabla_subselecciones"].round(3), use_container_width=True, hide_index=True)
    st.subheader("Gradiente por celda de la cuadrícula exploratoria principal 3×3")
    st.dataframe(D["tabla_gradientes_3x3"].round(4), use_container_width=True, hide_index=True)
    st.subheader("Resumen de las regiones finales")
    st.dataframe(R["tabla_regiones_finales"].round(3), use_container_width=True, hide_index=True)

with tab_hist:
    st.subheader("Histogramas normalizados")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for nombre, im in R["resultados_gris"].items():
        hist = np.bincount(im.ravel(), minlength=256)
        ax.plot(np.arange(256), hist / im.size, label=nombre, linewidth=1.3)
    ax.set(xlabel="Intensidad", ylabel="Frecuencia relativa", xlim=(0, 255), title="Distribución de intensidades")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.subheader("Funciones de distribución acumulada (CDF)")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for nombre, im in R["resultados_gris"].items():
        hist = np.bincount(im.ravel(), minlength=256)
        ax.plot(np.arange(256), np.cumsum(hist) / im.size, label=nombre, linewidth=1.4)
    ax.set(xlabel="Intensidad", ylabel="Probabilidad acumulada", xlim=(0, 255), ylim=(-0.02, 1.02), title="CDF de cada resultado")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with tab_local:
    idx = R["region_ejemplo_idx"]
    region_info = R["regiones_finales"][idx]
    st.subheader(f"Transformación local: {region_info['nombre']} · {region_info['padre']}")
    st.caption("Se muestra el recorte de la región final con mayor gradiente. Si se fusionó con vecinas, el histograma y el mapa T(r) corresponden al grupo fusionado.")
    st.image(R["region_ejemplo"], caption="Recorte local de entrada", use_container_width=True)
    h_entrada = R["hist_ejemplo"]
    h_salida = R["hist_salida_ejemplo"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    axes[0].plot(np.arange(256), h_entrada, linewidth=1.4)
    axes[0].set(title="Histograma local", xlabel="Intensidad", ylabel="Frecuencia")
    axes[0].grid(alpha=0.2)
    axes[1].plot(np.arange(256), R["cdfs_entrada_locales"][idx], label="CDF entrada")
    axes[1].plot(np.arange(256), R["cdfs_salida_locales"][idx], label="CDF salida V3")
    axes[1].set(title="CDF de la región", xlabel="Intensidad", ylabel="Probabilidad", ylim=(-0.02, 1.02))
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    axes[2].plot(np.arange(256), R["mapas_locales"][idx], label="T(r)")
    axes[2].plot(np.arange(256), np.arange(256), linestyle="--", alpha=0.55, label="Identidad")
    axes[2].set(title="Transformación local", xlabel="Intensidad de entrada", ylabel="Intensidad de salida", xlim=(0, 255), ylim=(0, 255))
    axes[2].grid(alpha=0.2)
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown("El mapa T(r) proviene de la transformación V3 calculada para el grupo regional correspondiente. Si varias regiones se fusionan, comparten el mismo mapa. Para la salida final, los mapas de los grupos se combinan mediante pesos inversamente proporcionales al cuadrado de la distancia a sus centros.")
    st.subheader("Diagnóstico de la región seleccionada")
    st.dataframe(R["tabla_diagnosticos_locales"].iloc[[idx]].round(4), use_container_width=True, hide_index=True)

with tab_download:
    st.subheader("Métricas globales")
    st.dataframe(R["tabla_metricas"].round(4), use_container_width=True)
    st.subheader("Diagnóstico de las 16 regiones")
    st.dataframe(R["tabla_diagnosticos_locales"].round(4), use_container_width=True, hide_index=True)
    if not R["historial_fusiones"].empty:
        st.subheader("Historial de fusiones adaptativas")
        st.dataframe(R["historial_fusiones"], use_container_width=True, hide_index=True)
    st.write(f"Fusión adaptativa: {'activada' if fusionar else 'desactivada'}. Grupos de transformación finales: {len(R['grupos'])}. Respaldo global: {'sí' if R['respaldo_imagen_completa'] else 'no'}.")
    dg = R["diagnostico_global"]
    st.write(f"V3 global — intervalo [{dg['r_min']}, {dg['r_max']}], A={dg['A']:.5g}, B={dg['B']:.5g}, C={dg['C']:.5g}, D={dg['D']:.5g}.")
    st.download_button("Descargar resultados e informes CSV en ZIP", data=_make_download_bundle(), file_name="resultados_V3v6_gradiente_jerarquico.zip", mime="application/zip", use_container_width=True)
    st.download_button("Descargar imagen V3v6 local", data=_png_bytes(R["resultados_visuales"][R["nombre_local"]]), file_name="imagen_V3v6_local.png", mime="image/png", use_container_width=True)
    st.download_button("Descargar métricas CSV", data=_csv_bytes(R["tabla_metricas"]), file_name="metricas_V3v6.csv", mime="text/csv", use_container_width=True)

st.divider()
st.caption("V3v6 experimental · Histograma V3 + regiones locales V3v2 + búsqueda jerárquica por gradiente en dos niveles + interpolación espacial suave. HE y CLAHE son referencias de comparación.")
