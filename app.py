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
