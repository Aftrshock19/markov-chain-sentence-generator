# Corpus audit — OPUS OpenSubtitles

Source: `es (1).txt`
Read-only. NFC + soft-hyphen/zero-width strip + whitespace collapse + single leading dash strip applied for evaluation only.

**Total lines:** 179,287,150

## Line char-length distribution (line-delimitation check)

| char-length bucket | lines | % |
|---|---|---|
| 0-40 | 138,399,208 | 77.19% |
| 41-80 | 33,583,361 | 18.73% |
| 81-160 | 6,792,126 | 3.79% |
| 161-320 | 508,570 | 0.28% |
| 321-640 | 3,752 | 0.00% |
| 641-1280 | 115 | 0.00% |
| 1281-5000 | 17 | 0.00% |
| 5001-inf | 1 | 0.00% |

## Per-predicate firing rates

| predicate | lines firing | % |
|---|---|---|
| has_space_token | 46 | 0.00% |
| legal_marker | 13,597 | 0.01% |
| subtitle_furniture | 1,354,633 | 0.76% |
| ocr_signature | 235,455 | 0.13% |
| high_digit_ratio | 1,159,515 | 0.65% |
| length_out_of_window | 73,143,071 | 40.80% |
| english_content_token | 188,212 | 0.10% |

## Co-occurrence

- firing ≥1 predicate (would-drop rate): 74,948,499 (41.80%)
- firing ≥2 predicates: 1,096,872 (0.61%)
- firing 0 predicates (estimated survival): 104,338,651 (58.20%)

## Token-count histogram

| tokens | lines | % |
|---|---|---|
| 1 | 22,837,378 | 12.74% |
| 2 | 24,730,777 | 13.79% |
| 3 | 25,194,433 | 14.05% |
| 4-7 | 67,028,132 | 37.39% |
| 8-15 | 33,482,345 | 18.68% |
| 16-30 | 5,633,602 | 3.14% |
| 31+ | 380,483 | 0.21% |

## english_content_token — which tokens fired

| token | line hits |
|---|---|
| the | 73,513 |
| you | 61,623 |
| and | 37,436 |
| of | 34,333 |
| that | 18,035 |
| for | 13,816 |
| your | 13,153 |
| this | 9,476 |
| with | 9,302 |
| miss | 1,465 |
| whole | 843 |
| Hail | 814 |

## Sampled firing lines per predicate (verbatim, judge false positives)

### has_space_token (46 firing; up to 15 random)

- L2817161: `SPACE COWBOYS`
- L120461618: `Bowie uso uno en "SPACE ODDITY".`
- L7390636: `SEE YA NEXT WEEK, SPACE COWBOYS!`
- L65174801: `Basada en DEAD SPACE de Visceral Games`
- L167857979: `PLATAFORMA DE LANZAMIENTO DE SPACE X`
- L116755507: `MISIÓN DE CONTROL, la NASA JOHNSON SPACE CENTER, HOUSTON, TEXAS 01:30`
- L65173221: `DEAD SPACE:`
- L24625817: `DEEP SPACE NINE`
- L24841556: `STAR TREK DEEP SPACE NINE`
- L68763740: `Jefa del Equipo 1 de Editorial Dou-Seh SPACE.`
- L167851771: `PLATAFORMA DE LANZAMIENTO DE SPACE X`
- L54548753: `HASTA EL PRÓXIMO EPISODIO, SPACE COWBOYS`
- L63859187: `HOT SPACE`
- L132075331: `SPACE PARANOIDS`
- L54545822: `SEE YA NEXT WEEK, SPACE COWBOYS`

### legal_marker (13,597 firing; up to 15 random)

- L105572179: `Ese derecho del Decreto Celestial,`
- L156512318: `Comprueba los CCTV de todas las caras.`
- L134860454: `LO SIENTO, FRIEDA, YO LO HE VISTO PRIMERO.`
- L129119738: `Camión Nº 1 a El Biar.`
- L61063816: `Señor Presidente, es ultrajante.`
- L74101782: `Señor Presidente, Los Ángeles fue devastado por una serie de tornados.`
- L70401700: `CENTRO ELECTORAL Nº 4`
- L161543312: `Nuestro último período de sesiones implicaba hacer una actuación en solitario en video.`
- L58875180: `Calle 4 de septiembre Nº 24.`
- L136613022: `CCTV ha sido inmovilizado.`
- L10182552: `Señor Presidente, tiene que sacar su culo de aquí.`
- L118140716: `Y que vive en el Nº 18 de la calle Jean-Jacques Rousseau.`
- L81555063: `Quién va a dibujar en un papel el diseño de nuestro Banco CCT.`
- L109187455: `¿Señor Presidente?`
- L59001935: `Estuvo en contacto con el Nº 6.`

### subtitle_furniture (1,354,633 firing; up to 15 random)

- L116823122: `SI ME AMAS, COMO DICES ...`
- L173413188: `SENTENCIA DEL TRIBUNAL MARCIAL`
- L138424638: `Busqué mi viejo atlas, lo tomé y me dije:`
- L527379: `El marinero Wiliam Bakewell recuerda el día de la partida:`
- L80450219: `Su batería bajó(disminuyó).`
- L173727960: `Sí, de seguro voy a llegar y le voy a decir:`
- L156020534: `Eso era de lo único que hablaban todos:`
- L54268080: `(suspira)`
- L49007656: `LUTHOR CORP FÁBRICA DE FERTILIZANTES Nº 3`
- L85377695: `Y Bart contestó:`
- L93663817: `*Tráeme mis flechas del deseo*`
- L106094937: `*Solo escucha el ritmo de una gentil bossa nova*`
- L130997866: `La última respuesta es la siguiente:`
- L104672543: `HAZ PASAR A LA SEÑORA MURPHY.`
- L47985032: `* Mirando a las dos y a las cuatro`

### ocr_signature (235,455 firing; up to 15 random)

- L22264025: `{\blur1.5}Transfieran todos los datos personales de la cápsula a la base de datos.`
- L70634974: `Con la misma capasidad de un 284-F36 de 50 kilos.`
- L165832986: `Cuando hombres como tú y los imbéciles del MI6 la CIA o la cloaca de donde hayan salido arrebatan no solo mi vida sino también las vidas inocentes de mi adorada esposa Natasha y de mi pequeño de seis años...`
- L5589908: `ChadItes`
- L152563800: `CerradIo.`
- L117007245: `{\ An4} TRANSFORMACIÓN`
- L47705946: `{\cH00FFFF}es el momento de que todos estemos unidos`
- L62439774: `Transmitiendo en vivo por WH2O--`
- L38887886: `¿Cómo no supieron los de Ml6 que tus padres fueron cosacos Lienz?`
- L151447550: `Un buen aplauso para nuestro más prominente munícipe o, munícipa o administradora municipal, FeIicia AIden.`
- L8769954: `¡R2, ve allí!`
- L125341748: `Sí, estoy hyperventiIating.`
- L136045632: `Estamos recibiendo información de que Liber8 tiene tres posibles objetivos de bombardeo.`
- L57704155: `No sé cómo cerrar todas las puertas de las zonas de seguridad, pero sí sé cómo ajustar la mezcla de oxígeno, nitrógeno y CO2 en sus cuartos, para que no se despierten si las alarmas se activan.`
- L82626053: `En este momento, mis colegas creen que estoy en eI consultorio... de algún hematóIogo, pero decidí venir aquí.`

### high_digit_ratio (1,159,515 firing; up to 15 random)

- L102436154: `15 dólares.`
- L40040535: `300 de caballería.`
- L88036529: `JUZGADO de DADE, 1970`
- L28560209: `Warp 5.`
- L50488521: `Smallville S08E22 "Doomsday"`
- L73355256: `Yo quería pelear 110.`
- L48404870: `En 13 meses.`
- L6800055: `8000.`
- L100778743: `Y media. 7:30.`
- L70445946: `Día Cinco 5:00`
- L59345472: `65`
- L39927969: `Volvemos al hospital, 17-45.`
- L14480494: `Clase, página 58.`
- L118390618: `Fue hace 30 años.`
- L136104411: `... 326 171.`

### length_out_of_window (73,143,071 firing; up to 15 random)

- L88466961: `Hola.`
- L156953722: `¡Oye, Jaco, levántate!`
- L117538625: `Skye.`
- L165758548: `De acuerdo.`
- L38804782: `Claro que sí.`
- L90456668: `Cambio de planes.`
- L1779605: `¿Dónde está Fischer?`
- L83166170: `Púdrete.`
- L37274353: `¡Rob, date prisa!`
- L127961700: `¿Asesinato?`
- L82798227: `Sí, lo sé.`
- L178311765: `Por qué?`
- L164851306: `¿Quién es?`
- L37424983: `Policía.`
- L165799381: `Escuchen.`

### english_content_token (188,212 firing; up to 15 random)

- L50826122: `Gracias a YYeTs "Garden of Eden Group" por la CC.`
- L102280579: `What would you say to $5000 to get us started?`
- L639684: `Un aplauso para "Timmy and the Lords of the Underworld!"`
- L67767070: `** Who's that girl?`
- L166048359: `# Like the Rolls Royce I can't have Your presence makes me scream #`
- L169575387: `If I could just die in your arms...`
- L124995634: `May I help you?`
- L141857428: `Pain-in-the-ax.`
- L19932904: `* all is dry * bright * 'round and 'round * the table...`
- L90828261: `'Us and Them' y 'The Great Gig in the Sky' es música de piano magnífica.`
- L6556016: `Entonces alguien ponía el "Ace of Spades" y nos quedábamos pensando...`
- L102102337: `Luckily I knew a chap in the loop - motor supplies.`
- L80597216: `Won't you, uh...`
- L73027726: `"Listen, I've traveled every road in this here land."`
- L150118456: `Didn't i say never to leave the cave?`

## Sampled lines firing NO predicate (would-keep set, judge false negatives)

- L57053336: `¿Se verá muy formal...`
- L38554464: `¡Langley, abre la puerta ahora mismo!`
- L60974055: `¿Puedes dejarme dormir hasta que salga el sol?`
- L16844072: `No puedo tocar en la tele.`
- L90332072: `Quería mantenerte fuera de todo esto.`
- L35610781: `Dije que los galos son invencibles.`
- L138425256: `Creo que te va a gustar.`
- L172127650: `Probablemente también las consumía.`
- L157859996: `Hemos venido demasiado pronto.`
- L106165421: `Amigo, confía en mí.`
- L149836074: `Hubiera preferido continuar así.`
- L74137138: `Todo viene de su amor por Bordeaux.`
- L76634470: `Cuando yo te ordeno hacer algo, ¿no tienes ganas de golpearme?`
- L15638927: `No estaba muy preocupado porque tenia seguro médico en otros países pero cuando me dijo que eran entre 23 y 24 mil pensé...`
- L125533171: `Es lo que diría un romántico.`

## Combined note

Estimated survival (lines firing zero predicates): **58.20%** (104,338,651 of 179,287,150).
