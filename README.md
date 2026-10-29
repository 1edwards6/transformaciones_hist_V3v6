# V3v6 — Mejoramiento matemático de imágenes con cuadrícula jerárquica por gradiente

Aplicación de Streamlit construida a partir del núcleo matemático de la V3 (reorganización del histograma), las transformaciones locales de la V3v2 y la selección adaptable por gradiente.

## ¿Qué hace la V3v6?

1. **V3 global:** calcula un histograma objetivo a partir del intervalo mínimo de intensidades que cubre el porcentaje seleccionado. Aproxima la distribución con PCHIP, interpolación lineal o una aproximación trigonométrica basada en Fourier; ajusta el área para conservar el número de píxeles y reasigna las intensidades de manera ordenada.
2. **Selección del primer nivel:** explora particiones candidatas 3×3, puntúa la región central frente a las ocho regiones exteriores y coloca los ejes principales 2×2 en el punto medio de la región candidata.
3. **Selección del segundo nivel:** repite una búsqueda por gradiente dentro de cada uno de los cuatro cuadrantes. Cada cuadrante se divide en cuatro celdas, para un máximo de **16 regiones finales**.
4. **Transformaciones locales:** calcula el histograma y el mapa V3 por región o por grupo de regiones fusionadas.
5. **Suavizado espacial de mapas:** combina los cuatro mapas de grupo más cercanos mediante pesos inversamente proporcionales al cuadrado de la distancia a sus centros. Es una generalización para regiones irregulares; no es una interpolación bilineal regular como la malla 2×2 anterior.
6. **Comparación:** presenta la imagen original, V3 global, V3v6 local, ecualización clásica HE y CLAHE, además de métricas, histogramas, CDF y diagnósticos exportables.

## Archivos

- `app.py`: interfaz Streamlit.
- `modelo_v3.py`: funciones de la V3 y procesador de la V3v6.
- `requirements.txt`: dependencias del entorno.
- `referencia/Transformacion_Histograma_V3v4_Gradiente_2x2.ipynb`: cuaderno fuente proporcionado, conservado para trazabilidad.
- `tests/test_modelo.py`: pruebas básicas con imágenes sintéticas.

## Ejecutar localmente

Recomendado: Python 3.10–3.12.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Instala las dependencias e inicia la app:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Desplegar en Streamlit Community Cloud

1. Descomprime el ZIP.
2. Sube `app.py`, `modelo_v3.py`, `requirements.txt` y `README.md` a la raíz de un repositorio GitHub. La carpeta `referencia/` y `tests/` son opcionales para ejecutar la app, pero se conservan como documentación.
3. En Streamlit Community Cloud crea una app desde ese repositorio y selecciona `app.py` como archivo principal.

## Controles relevantes

- **Paso de búsqueda:** menor paso explora más candidatos y puede tardar más.
- **Suavizado σ:** reduce la sensibilidad al ruido al calcular el gradiente.
- **Peso de exterior:** controla la penalización de los gradientes fuera de la celda central candidata.
- **Fusión adaptativa:** permite compartir un mapa de transformación entre regiones adyacentes con desviación estándar inferior al umbral.
- **Protección de regiones constantes:** evita alterar artificialmente regiones que solo tienen un nivel de intensidad.

## Notas y límites experimentales

- “Hasta 16 regiones” describe las celdas geométricas de la subdivisión de dos niveles. Si se activa la fusión adaptativa, varias celdas pueden compartir un mapa de transformación y el número de grupos de mapas será menor.
- Para recortes locales demasiado pequeños, la selección secundaria utiliza el punto medio geométrico como respaldo.
- La interpolación espacial basada en centros busca reducir discontinuidades, pero debe evaluarse empíricamente en imágenes reales; no garantiza eliminarlas por completo.
- HE y CLAHE son referencias de comparación, no una afirmación de que una métrica concreta equivalga por sí sola a mejor calidad visual.
