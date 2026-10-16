# Calibración con clips reales

Los umbrales de `DEFAULT_THRESHOLDS` (en `tactical_analysis.py`) empezaron como valores
heurísticos. Esta calibración los compara con lo que una persona ve en clips reales de
transmisión de TV, con el mismo modo **rápido** que usa la app (`frame_stride` 5, `max_frames` 120).

Es una calibración **orientativa**: 7 clips cortos y 1 persona etiquetando. Sirve para corregir
umbrales que claramente no funcionaban con cámara de TV, no para afirmar una precisión general.

## Cómo se hizo

1. `python -m calibration.calibrate measure calibration/clips/*.mp4` analiza cada clip con el
   pipeline real y guarda, por equipo, el valor de cada indicador en cada cuadro
   (`calibration/measurements.json`).
2. Una persona respondió, por clip y por equipo (identificado por el color de la camiseta),
   preguntas de sí o no. Las respuestas están en `calibration/labels.json`:

   | Pregunta | Reglas que la revisan |
   |---|---|
   | ¿Mucho espacio entre defensa y mediocampo? | `defensive_midfield_distance`, `largest_line_gap` |
   | ¿Equipo desordenado o poco compacto? | `team_compactness_area`, `average_player_distance`, `nearest_teammate_distance`, `isolated_players`, `network_components` |
   | ¿Nadie presiona al que tiene el balón? | `pressure_on_ball` |
   | ¿Amontonado por el centro, sin abrir la cancha? | `attacking_width`, `team_width`, `max_zone_concentration` |

3. `python -m calibration.calibrate evaluate` cuenta, para cada regla, aciertos, falsas alarmas
   (la app muestra un problema que la persona no vio) y problemas no detectados, y busca el
   umbral que más coincide. Una regla "se dispara" igual que en `tactical_insights.py`: al menos
   `MIN_SAMPLES` cuadros y el umbral cruzado en al menos `MIN_PERSISTENCE` de ellos.

Los clips no se guardan en el repositorio (son de transmisiones de TV y pesan mucho): solo los
valores medidos y las etiquetas. `calibration/clips/` está en `.gitignore`.

## Clips

| Clip | Equipo A | Equipo B | Problemas que vio la persona |
|---|---|---|---|
| Francia – Marruecos | oscuro | rojo | A: líneas separadas, poco compacto, amontonado. B: no presiona |
| Ataque | blanco | oscuro | A: no presiona, amontonado. B: líneas separadas, poco compacto, amontonado |
| Contraataque por banda | blanco | rojo | A: ninguno. B: todos |
| Contraataque y mala defensa | rojo | blanco | A: ninguno. B: todos |
| Pase filtrado a gol | rojo oscuro | blanco | A: líneas separadas, poco compacto, no presiona. B: amontonado |
| Presión | oscuro | amarillo | A: poco compacto. B: líneas separadas, poco compacto, no presiona |
| Tiro de esquina a gol | blanco | rojo | Ambos: amontonados en el área (es normal en un córner) |

## Lo que se encontró

### 1. El seguimiento descartaba más de la mitad de los jugadores

Al medir, varios equipos tenían menos de 8 cuadros con datos y las reglas no podían decir nada.
La causa no eran los umbrales sino el seguimiento: ByteTrack solo confirma una pista nueva cuando
la vuelve a encontrar en el cuadro siguiente, y con la cámara de TV moviéndose entre cuadros
muestreados (cada 5) muchas pistas nunca se confirmaban.

| Clip | Jugadores detectados por cuadro | Usados antes | Usados ahora |
|---|---|---|---|
| Francia – Marruecos | 12,5 | 4,9 | 12,5 |
| Ataque | 15,0 | 5,8 | 15,0 |
| Contraataque por banda | 10,5 | 3,8 | 10,5 |
| Contraataque y mala defensa | 9,1 | 4,6 | 9,1 |
| Pase filtrado a gol | 6,9 | 2,1 | 6,9 |
| Presión | 13,0 | 3,9 | 13,0 |
| Tiro de esquina a gol | 12,6 | 7,6 | 12,6 |

Cambio: los jugadores que el tracker no devuelve se usan igual en la foto táctica de ese cuadro,
con un id negativo (nunca se confunde con una pista). Su equipo sale del color de camiseta más
cercano a los dos colores de equipo, que se siguen calculando solo con pistas (muchas muestras,
más estables). Si el color no se parece a ninguno (árbitro), queda sin equipo.

### 2. Un árbitro junto al balón rompía el análisis

Con más jugadores visibles apareció un error real: si la persona más cercana al balón era un
árbitro (sin equipo), el cálculo de posesión fallaba con `KeyError`. Ahora solo un jugador de un
equipo puede tener el balón. Hay una prueba para esto en `tests/test_indicators.py`.

### 3. Fase del juego: la forma se juzga mientras el equipo defiende

Con todos los jugadores visibles, las falsas alarmas subieron (de 6 a 13) por una razón clara: las
reglas de forma no sabían si el equipo atacaba o defendía. En los dos contraataques, el equipo que
contraataca (que la persona vio bien) salía "partido en grupos" o con "líneas separadas", cuando
al atacar rápido abrirse es lo correcto.

Cambio (`INDICATOR_PHASE` en `tactical_analysis.py`):

- **Posesión con memoria.** El balón se ve en el 14–52 % de los cuadros según el clip. Un equipo
  conserva la posesión 2 segundos después del último cuadro en que uno de sus jugadores estaba
  junto al balón, o hasta que se ve al rival con él (`possession_by_frame`).
- **Forma defensiva** (líneas, compactación, distancia al compañero, aislados, grupos, zona más
  ocupada): se mide **solo en los cuadros en que el rival tiene el balón**.
- **Amplitud** (`team_width`): se mide solo en los cuadros en que el equipo **no** está defendiendo.
- Un equipo que casi no defendió en el video **no se evalúa** en su forma defensiva, y el análisis
  lo dice en sus avisos, en vez de darlo por bueno.
- La evidencia de cada alerta dice en qué momentos se midió.

Se probaron otras variantes con las mismas etiquetas antes de elegir esta:

| Variante | Aciertos | Falsas alarmas |
|---|---|---|
| Todos los cuadros (etapa anterior) | 23 | 13 |
| Forma en cuadros sin balón propio (defendiendo o sin saber) | 20 | 13 |
| **Forma solo defendiendo + amplitud sin defender** | **20** | **3** |

### 4. Umbrales

Un umbral solo se cambió si encontraba **más** problemas de los que vio la persona **sin agregar
falsas alarmas** y sin perder los que ya encontraba (`best_threshold` en `calibrate.py`), ya
midiendo cada regla en su fase.

| Indicador | Antes | Ahora | Por qué |
|---|---|---|---|
| `team_compactness_area` | 0,25 | 0,12 | Con 0,25 no se disparó en ningún clip: la cámara de TV muestra solo parte del equipo, así que el área visible siempre es pequeña. |
| `nearest_teammate_distance` | 0,15 | 0,115 | Acierta en 4 equipos en vez de 1, sin falsas alarmas. |
| `average_player_distance` | 0,30 | 0,28 | Un acierto más, sin falsas alarmas. |
| `defensive_midfield_distance` | 0,22 | 0,18 | Dos aciertos más, sin falsas alarmas (solo 4 casos medibles: es el cambio menos respaldado). |

También se bajó la confianza fija de dos reglas que fallaron más con los clips reales:
"zona concentrada" (0,85 → 0,6; 2 falsas alarmas de 7) y "amplitud insuficiente" (0,75 → 0,65;
1 de 3). Así la app las muestra con menos confianza que las demás.

No se cambiaron `network_components`, `isolated_players`, `max_zone_concentration` ni `team_width`
(ningún valor mejoraba sin agregar falsas alarmas o perder aciertos). `attacking_width`,
`largest_line_gap` y `pressure_on_ball` tuvieron un solo caso medible: no alcanza para calibrar.

### 5. Experimento descartado: buscar el balón en dos mitades

De los 27 problemas que la app no encuentra, 16 son de equipos que no se pudieron medir, casi
siempre porque no se vio el balón (sin balón no se sabe quién defiende). El modelo trabaja con
imágenes de 640 px: un cuadro de 1920 px se reduce 3 veces y el balón queda de pocos píxeles.

Se probó buscarlo de nuevo en las dos mitades del cuadro cuando no aparece en el cuadro entero
(`precise_ball`, `--precise-ball` en `calibrate.py`). El balón se vio el doble de veces (del 28 %
al 60 % de los cuadros, en promedio) y la mayoría de lo encontrado era el balón real, pero:

| Modo | Aciertos | Falsas alarmas |
|---|---|---|
| Normal | 20 | 3 |
| Balón en dos mitades | 24 | 18 |
| Balón en dos mitades + posesión solo si un equipo está claramente más cerca | 20 | 8 |

Con más cuadros con balón aparecen más momentos disputados, donde el jugador más cercano al
balón es el defensor que presiona y la posesión se asigna al equipo equivocado. Como los datos
empeoran, **la app no lo usa**: queda apagado y documentado para repetir el experimento.

Lo que sí se conservó: el balón debe tener **césped alrededor** (`_ball_on_pitch`). En un clip,
el escudo de un equipo en el marcador de la TV se detectaba como balón. Con el filtro el resultado
del modo normal no cambia (20 aciertos, 3 falsas alarmas) y se evita ese error.

## Resultado

Por pregunta y equipo (14 equipos, 96 respuestas en total), comparando lo que la app muestra con
lo que vio la persona:

| | Problemas encontrados | Falsas alarmas | Cuando avisa, acierta | Coincidencia total |
|---|---|---|---|---|
| Original (seguimiento y umbrales iniciales) | 14 de 47 | 6 | 70 % | 57 de 96 |
| Etapa 1: todos los jugadores + umbrales | 23 de 47 | 13 | 64 % | 59 de 96 |
| **Etapa 2: fase del juego + umbrales** | **20 de 47** | **3** | **87 %** | **66 de 96** |

Lo más importante para un cuerpo técnico es la cuarta columna: **cuando la app avisa un problema,
acertó 20 de 23 veces**. Encuentra 20 de los 47 problemas que vio la persona; la mayoría de los que
no encuentra son de equipos que casi no defendieron en el clip, y para esos la app avisa que no se
evaluaron.

## Limitaciones que quedan

- **Pocos datos.** 7 clips cortos (de 10 a 20 segundos) y una sola persona. Los porcentajes son
  una medida honesta en estos clips, no una garantía general. Con más clips (y más de una persona
  etiquetando, idealmente alguien del cuerpo técnico) los umbrales se recalculan con los mismos
  dos comandos.
- **El balón se ve poco** (modelo general, balón pequeño en plano abierto). Sin balón no se sabe
  quién ataca, y esos cuadros no cuentan para la forma defensiva. Verlo más (experimento 5) no
  basta: también hace falta saber mejor quién lo tiene en las jugadas disputadas.
- **Las etiquetas son por clip, no por momento.** La persona juzgó a cada equipo en el clip
  completo; si un equipo defendió mal unos segundos y luego contraatacó bien, la etiqueta no lo
  distingue. Etiquetas por momento permitirían medir mejor las fases.
- **Zona concentrada** sigue siendo la regla menos confiable: la cámara sigue al balón y los
  jugadores visibles están, por construcción, cerca unos de otros.
- **Sin homografía**, las distancias son relativas a la imagen y cambian con el zoom de la cámara.
  Con una cámara fija que vea toda la cancha se podrían medir en metros.
