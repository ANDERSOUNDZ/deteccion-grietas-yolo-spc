# Detección de daño estructural por grietas (YOLO + Control Estadístico de Procesos)

Proyecto de la **Evaluación 2** de *Modelos Matemáticos y Simulación* (UNIB.E).

El sistema recibe la foto de una pared o superficie, **segmenta las grietas con YOLO** (Ultralytics),
mide cuánto daño hay y lo clasifica en **BAJO / MEDIO / ALTO** usando **Control Estadístico de Procesos (SPC)**.
Además estima la **probabilidad de falla** y proyecta el daño a 12 meses en distintos escenarios
(envejecimiento normal, construcción cercana, movimiento de suelo y sismo).

## Cómo funciona

```
Foto -> YOLO11n-seg (segmenta la grieta) -> medición (% de área, longitud, ancho)
     -> carta de control SPC -> nivel BAJO / MEDIO / ALTO
     -> probabilidad de falla (lognormal) y escenarios a 12 meses -> recomendación
```

| Fase | Qué hace |
|---|---|
| I. Dataset | Descarga Crack-Seg (4 028 imágenes: 3 717 train / 200 val / 111 test) y convierte las etiquetas a formato YOLO |
| II. Entrenamiento | *Fine-tuning* de `yolo11n-seg` (30 épocas, `imgsz=416`, `batch=16`) en GPU T4 de Colab |
| III. Modelo matemático | Mide la grieta, define el proceso base, carta de control, niveles de daño, Cpk, probabilidad de falla y escenarios |
| IV. Validación | Compara el sistema contra las etiquetas reales de las 111 imágenes de prueba y hace análisis de sensibilidad |
| Demo | Interfaz Gradio: subes una imagen y recibes el diagnóstico |

### Niveles de daño (regla SPC sobre el % de área con grieta)
Con el proceso base de las imágenes de entrenamiento (promedio 2.16 %, σ 1.39 %):

| Nivel | Regla | Acción sugerida |
|---|---|---|
| BAJO | área ≤ promedio (2.16 %) | aprobar y monitorear |
| MEDIO | promedio < área ≤ promedio + 2σ (4.94 %) | sellar y reinspeccionar |
| ALTO | área > promedio + 2σ | rechazar / evaluación urgente |

Límite de especificación: una pieza **falla** si la grieta cubre más del 5 % de la imagen (USL = 5 %).

## Resultados (conjunto de prueba, máscara de la grieta)

| Métrica | Valor |
|---|---|
| Precisión | 0.846 |
| Recall | 0.592 |
| mAP@0.5 | 0.625 |
| mAP@0.5:0.95 | 0.225 |

- Proceso: Cpk = 0.68 (no capaz), probabilidad de falla del proceso 4.4 %.
- Probabilidad de falla a 12 meses: normal 7.0 %, construcción cercana 15.3 %, movimiento de suelo 29.4 %, sismo 43.5 %.
- Validación: correlación 0.65 entre el área real y la medida por YOLO; el nivel (BAJO/MEDIO/ALTO) acierta el 72 %.

## Cómo ejecutarlo (Google Colab)

1. Abre `deteccion_grietas_yolo_spc.ipynb` en Google Colab y activa **GPU T4** (Entorno de ejecución > Cambiar tipo de entorno).
2. Descarga el dataset **Crack-Seg** (archivo `.ndjson`) desde la plataforma de Ultralytics. Las URLs de las imágenes dentro del archivo **caducan** en aproximadamente una semana.
3. El notebook lee el archivo en dos partes (`crack_p1.ndjson` y `crack_p2.ndjson`) porque así se subió a Colab. Para dividirlo:
   ```python
   lineas = open("crack-seg.ndjson", encoding="utf-8").read().splitlines()
   mitad = len(lineas) // 2
   open("crack_p1.ndjson", "w", encoding="utf-8").write("\n".join(lineas[:mitad]) + "\n")
   open("crack_p2.ndjson", "w", encoding="utf-8").write("\n".join(lineas[mitad:]) + "\n")
   ```
   Sube ambos archivos al panel de archivos de Colab.
4. Ejecuta todas las celdas. El entrenamiento tarda unos 10 a 25 minutos.

Dependencias (ver `requirements.txt`): `ultralytics`, `gradio`, `scikit-image`, `scipy`, `seaborn`, `pandas`, `matplotlib`, `opencv-python`, `requests`.

## Limitaciones (importante)

- Los **3 niveles son una regla sobre el % de grieta medido**, no etiquetas hechas por ingenieros.
- El nivel **ALTO** casi no aparece en el conjunto de prueba (solo 2 imágenes), por lo que su acierto no es significativo.
- Los **factores de crecimiento** de los escenarios (sismo, suelo, construcción) son **supuestos del modelo**: el dataset no contiene información de tiempo ni de sismos. Deben calibrarse con inspecciones periódicas.
- El dataset son **calles y paredes**, no cimientos.
- Se mide en píxeles y en % de la imagen, no en milímetros reales (no hay escala).
- Falsos positivos (sombras, manchas, juntas) y falsos negativos (grietas muy finas) afectan la medición y, con ella, la incertidumbre del modelo.
- En el entrenamiento, con `optimizer="auto"` YOLO eligió AdamW (lr = 0.002) e ignoró el `lr0=0.01` indicado en el código.

## Archivos

- `deteccion_grietas_yolo_spc.ipynb`: notebook completo (Fases I a IV y demo).
- `requirements.txt`: dependencias.
- `.gitignore`: excluye datos, entrenamientos y pesos base.

El archivo de pesos entrenados `best.pt` (≈ 6 MB) se genera en el PASO 16 del notebook.

## Créditos

- Dataset: *Crack-Seg*, curado por Ultralytics (fuente: dataset "crack", 2022).
- Modelo: Ultralytics YOLO11.
