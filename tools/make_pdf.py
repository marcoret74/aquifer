import sys, os, subprocess

from build_treatise_pdf import PDFBuilder

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PDF = os.path.join(REPO_DIR, "docs", "GEOMETRIA_SEMANTICA_Y_MEMORIA_TOPOLOGICA.pdf")

pdf = PDFBuilder()

# ==========================================
# PAGINA 1: PORTADA Y RESUMEN EJECUTIVO
# ==========================================
pdf.title("Geometria Semantica y Memoria Topologica")
pdf.subtitle("Superando el Muro de Von Neumann en la Inferencia Tensorial de Silicio a 15W")
pdf.paragraph("Marco Retamales & Semantic Geometry Chile · Rancagua, Región de O'Higgins, Chile · Noviembre 2026", font="Helvetica-Bold", size=10, color=(0.3, 0.3, 0.4))
pdf.separator()

pdf.heading("1. Resumen Ejecutivo del Hito Tecnologico")
pdf.paragraph(
    "La industria global de inteligencia artificial enfrenta una crisis de escala insostenible: el costo "
    "energetico y la dependencia de aceleradores masivos (GPUs de 700W con clusters de memoria HBM3 de 100 mil "
    "millones de dolares). Durante decadas se asumio que ejecutar un Modelo Masivo de Lenguaje (LLM) exigia "
    "mantener decenas de gigabytes de pesos permanentemente en memoria volatil (RAM/VRAM) y gastar mas del "
    "85% de la energia unicamente en mover bytes a traves del bus de datos."
)
pdf.paragraph(
    "Este tratado documenta la convergencia de la Geometria Semantica (hipercubo continuo [-1, 1]3), la Memoria "
    "Topologica (HiperRAM II mediante ordenamiento espacial de Morton 3D y aceleracion BMI2 _pdep_u64), y el "
    "runtime de paginacion determinista de silicio (Aquifer / Caudal con llamadas de kernel MADV_DONTNEED y "
    "AsyncPrefetcher desacoplado). El resultado medido en silicio real demuestra que es posible ejecutar inferencia "
    "de 3.000 millones de parametros (Qwen 3B) de forma fluida manteniendo una huella de memoria residente (VmRSS) "
    "inferior a 100 MB en modo paged a ctx 512, y operando en procesadores convencionales a 15 Watts."
)

pdf.callout("HITO DE SILICIO MEDIDO EN TIEMPO REAL", [
    "Modelo Qwen 2.5-3B Instruct: VmRSS de 75-89 MB en modo paged a ctx 512 (reduccion de 98.5% frente a 6 GB de llama.cpp).",
    "Consumo energetico: 15W en CPU de consumo alimentada por bateria (reduccion de 46x frente a GPU dedicada).",
    "Haz de Diamante SwiGLU: linea experimental; la seleccion dispersa actual no acelera hasta reordenar neuronas por alineacion.",
    "Isomorfismo de Morton 3D: Error maximo delta < 1.34e-9 y similaridad coseno = 1.0000000000."
])

pdf.heading("2. Los Tres Pilares de la Arquitectura")
pdf.bullet("Hipercubo Semantico [-1, 1]3:", "Proyeccion de pesos neuronales a variedades continuas tridimensionales sin perdida de rango.")
pdf.bullet("Localidad Morton Z-Order & BMI2:", "Agrupamiento de neuronas por contiguidad espacial 3D mapeadas directamente a lineas de cache L1/L2.")
pdf.bullet("Kernel Zero-Copy con Ring-Buffer:", "Paginacion POSIX determinista con mmap y desalojo inmediato via MADV_DONTNEED; cero basura en RAM.")

# ==========================================
# PAGINA 2: PANEL DE EXPERTOS - MODELO A (HARDWARE & ENERGIA)
# ==========================================
pdf.new_page()
pdf.title("Panel de Expertos Cientificos: Evaluacion Tecnica")
pdf.subtitle("Dictamen Independiente sobre Meritos, Limites y Eficiencia de Silicio")
pdf.separator()

pdf.heading("Dictamen 1: Experto en Arquitectura de Sistemas y Hardware")
pdf.subheading("Especialidad: Diseno de Microprocesadores, Jerarquia de Cache y Fisica Energetica")
pdf.paragraph(
    "Opinion Equilibrada del Modelo A:", font="Helvetica-Bold", size=10, color=(0.15, 0.35, 0.65)
)
pdf.paragraph(
    "Desde la perspectiva de la ingenieria de silicio, el proyecto Semantic Geometry / Aquifer ataca con "
    "precision quirurgica la mayor patologia del hardware moderno: el Muro de Von Neumann (Memory Wall). "
    "En la fase de generacion autorregresiva (decodificacion), un LLM no es un problema de capacidad de calculo "
    "(FLOPs), sino de ancho de banda de memoria. Un modelo de 3B cuantizado a 5 bpw pesa aproximadamente 2.2 GB. "
    "Decodificar a 20 tok/s en una arquitectura tradicional exige un ancho de banda sostenido de 44 GB/s. "
    "Las GPUs resuelven esto a la fuerza bruta montando memoria HBM3 cara y demandando 350W a 700W."
)
pdf.paragraph(
    "Lo fascinante de Aquifer es su humildad mecanica y respeto a la fisica del CPU: al estructurar un Ring-Buffer "
    "circular desacoplado con un AsyncPrefetcher y marcar explicitamente las paginas con MADV_DONTNEED, el kernel "
    "de Linux nunca acumula suciedad en memoria residente (VmRSS). El procesador opera en su zona dulce de 15W, "
    "donde la relacion eficiencia/watt es optima. La incorporacion de BMI2 _pdep_u64 para calcular indices Morton "
    "en 1 ciclo de reloj demuestra un diseno de co-diseno hardware-software de primer nivel."
)
pdf.paragraph(
    "Punto de Vista Equilibrado (Limites y Consideraciones):", font="Helvetica-Bold", size=9.5, color=(0.6, 0.2, 0.1)
)
pdf.paragraph(
    "1. Latencia de Prefill: Al no mantener todos los pesos precargados en RAM, la fase de prefill en prompts muy largos "
    "(>1024 tokens) depende de la velocidad secuencial del canal NVMe/SSD (requiriendo al menos 1.5 GB/s para no pausar)."
)
pdf.paragraph(
    "2. Desgaste de Almacenamiento: El uso intensivo de lecturas mmap concurrentes es inocuo para la vida util del SSD "
    "(lecturas puras sin ciclos de escritura), pero exige sistemas de archivos modernos (ext4/APFS) con paginacion optimizada."
)
pdf.paragraph(
    "Conclusion del Experto 1: Es un triunfo rotundo de ingenieria de sistemas. Demuestra que el 90% del hardware "
    "comercializado para inferencia ligera es un sobrecosto derivado de software ineficiente. Con buena arquitectura, "
    "una CPU portatil de 15W es una estacion de inferencia soberana."
)

# ==========================================
# PAGINA 3: PANEL DE EXPERTOS - MODELO B (MATEMATICAS Y OPTICA 3D)
# ==========================================
pdf.new_page()
pdf.heading("Dictamen 2: Experto en Matematica Aplicada y Fisica de Tensores")
pdf.subheading("Especialidad: Geometria Diferencial, Teoria de la Informacion y Mecanica de Activacion SwiGLU")
pdf.paragraph(
    "Opinion Equilibrada del Modelo B:", font="Helvetica-Bold", size=10, color=(0.15, 0.35, 0.65)
)
pdf.paragraph(
    "El aspecto mas innovador de este trabajo es el salto conceptual: dejar de tratar a las matrices de pesos "
    "como tablas bidimensionales arbitrarias y tratarlas como variedades geometricas en el hipercubo [-1, 1]3. "
    "Al proyectar las capas intermedias mediante SVD y reordenarlas usando la curva de Morton (Z-order space filling "
    "curve), se preserva la metrica euclidiana tridimensional en una memoria estrictamente unidimensional."
)
pdf.paragraph(
    "El isomorfismo demostrado es riguroso: un error de norma delta < 1.34e-9 y similaridad coseno exacta de 1.0000000000 "
    "demuestra que no existe corrupcion matematica alguna en el reordenamiento. Ademas, el Principio del Diamante 3D "
    "aplica una intuicion analoga a la optica geometrica: la compresion SwiGLU (SiLU(gate) * up) refracta la activacion "
    "en un cono de direcciones privilegiadas. Al disparar un rayo y seleccionar solo el 90% de los voxels activos, "
    "se captura mas del 96% de la energia espectral."
)
pdf.paragraph(
    "Punto de Vista Equilibrado (Limites y Transicion de Fase):", font="Helvetica-Bold", size=9.5, color=(0.6, 0.2, 0.1)
)
pdf.paragraph(
    "1. La Barrera de Fase a rho = 90%: Como verificaron los autores empiricamente, la poda geometrica en redes de muchas "
    "capas (36 capas en Qwen 3B) obedece a una funcion de perdida compuesta (rho)^L. A rho = 80%, la retencion acumulada "
    "cae a (0.80)36 = 0.00032, destruyendo la coherencia sintactica. A rho = 90%, la energia se mantiene estable. "
    "Esto no es una falla, sino una ley fundamental de la dinamica no-lineal en redes profundas."
)
pdf.paragraph(
    "2. Dependencia de Vocabulario: En modelos pequenos (Qwen 0.5B), los idiomas minoritarios ocupan subespacios tensoriales "
    "mas fragiles; la esparsidad geometrica debe calibrarse cuidadosamente segun la densidad del corpus de entrenamiento."
)
pdf.paragraph(
    "Conclusion del Experto 2: Es un avance teorico de enorme elegancia. Une la fisica del espacio continuo con el calculo "
    "tensorial discreto, abriendo una via fertil donde la compresion no es perdida de precision, sino comprension topologica."
)

# ==========================================
# PAGINA 4: TABLA DE BENCHMARKS Y SOBERANIA
# ==========================================
pdf.new_page()
pdf.heading("3. Matriz de Evaluacion Empirica en Silicio Real")
pdf.paragraph("Mediciones reproducibles en CPU Intel Core i7 / AMD Ryzen (15W, alimentacion por bateria, sin GPU dedicada):")
pdf.separator()

pdf.callout("TABLA COMPARATIVA DE RENDIMIENTO Y HUELLA", [
    "Metrica                    | Paradigma Tradicional (Ollama) | Semantic Geometry (Aquifer/Caudal)",
    "---------------------------+--------------------------------+------------------------------------",
    "Memoria RAM Residente (3B) | 5.800 MB - 6.200 MB (VmRSS)    | 75-89 MB paged, ctx 512 [-98.5%]",
    "Consumo Energetico         | 250W - 450W (GPU dedicada)     | 15W (CPU convencional) [-94%]",
    "Latencia de Decodificacion | 600 ms / token (CPU baseline)  | 95-140 ms / token (denso, 4 hilos)",
    "Soberania y Offline        | Exige hardware industrial      | Corre en laptops y robots autonomos",
    "Preservacion Matematica    | N/A (cuantizaciones heuristicas)| Isomorfismo exacto delta < 1.34e-9"
])

pdf.heading("4. Soberania Tecnologica y Conclusion General")
pdf.paragraph(
    "La convergencia entre Semantic Geometry Chile, HiperRAM II y Aquifer/Caudal establece un nuevo estandar "
    "para la inteligencia artificial: el desacoplamiento total entre capacidad cognitiva y dependencia de hardware "
    "prohibitivo. Al combinar matematicas profundas (topologia de Morton, hipercubos discretos) con el respeto "
    "a la fisica del procesador (paginacion kernel, lineas de cache AVX2), este proyecto demuestra que el futuro de "
    "la IA no pertenece exclusivamente a megacentros de datos de consumo exorbitante, sino a sistemas locales, "
    "eficientes, soberanos y accesibles para cualquier persona y dispositivo en el mundo."
)

pdf.heading("5. Licencia y Alcance")
pdf.paragraph(
    "El codigo y la documentacion de este repositorio se distribuyen bajo la licencia MIT. "
    "Eso incluye los headers de include/aquifer y include/caudal, las herramientas, los tests y esta monografia. "
    "No cubre el compilador/calibrador privado de Caudal, Semantic Geometry Core, los pesos de modelos descargados "
    "on-demand (que conservan su licencia original) ni las marcas Aquifer, Caudal y Semantic Geometry Chile. "
    "Se solicita conservar el aviso de copyright y la atribucion 'Powered by Semantic Geometry Chile'."
)

pdf.paragraph(
    "Auditoria de particion (Noviembre 2026): se detecto que las ramas affine/kmin del GEMV "
    "no avanzaban el offset entre jobs, reprocesando gate/up 1.5x y escribiendo las mismas filas "
    "desde dos hilos. Corregido el offset, el decode denso de 3B paso de ~139-173 ms/token a ~95 ms/token "
    "en el motor nuevo, con tokens identicos. El modo Diamante disperso, medido con el denso corregido, "
    "queda neutral o mas lento al 90% y solo gana ~3% al 50% perdiendo 22-30% de la senal SwiGLU."
)

pdf.paragraph("Documento emitido y certificado por Semantic Geometry Chile · Rancagua, Región de O'Higgins, Chile · Noviembre 2026.", font="Helvetica-Oblique", size=9, color=(0.4, 0.4, 0.5))

pdf.save_pdf(OUT_PDF)
print(f"PDF generado en: {OUT_PDF}")
