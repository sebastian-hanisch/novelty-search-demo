# Novelty Search – suchen, ohne das Ziel zu kennen (Streamlit-Demo)

---

Interaktive Demo zu **Novelty Search**. **Zweites Stück der Konzepte-Linie „Novelty Search und Quality-Diversity“** im Portfolio von
[Sebastian Hanisch](https://sebastianhanisch.net) (Operations Research und Machine Learning): ein Verfahren, ein wachsendes Beispiel, jedes Folgestück hebt genau eine Annahme auf.

Derselbe Hof-Roboter, dasselbe täuschende Labyrinth und derselbe Genetische Algorithmus wie im ersten Stück ([`deception-maze-demo`](https://github.com/sebastian-hanisch/deception-maze-demo)), nur wählt die Suche nicht mehr nach dem Abstand
zum Ziel aus, sondern nach **Novelty**: Ausgewählt wird, wessen Endposition **anders** ist als die der anderen. Das Ziel kommt in der Auswahl nicht mehr vor. Das Stück misst, **wann das die Falle löst**, was ein **Archiv** der bisherigen
Verhalten beiträgt (hier: nichts) und **wo Novelty Search scheitert** (Stufe 4, auch mit dem Zehnfachen des Budgets).

## Kernfrage

Löst die Auswahl nach Neuartigkeit, was die Auswahl nach Zielnähe nicht schafft, und was davon hängt am Archiv, an k und an der Population?

## Modell und Methodik

- **Labyrinth und Roboter** (`ns_maze.py`, `ns_robot.py`): unverändert aus Stück 1. 24 × 24 Zellen, Start (3.5, 3.5), Ziel (3.5, 20.5); Stufe 0 freies Feld, Stufen 1 bis 3 eine Querwand mit Öffnung links, in der Mitte oder rechts, Stufe 4 drei Wände
  (Öffnungen rechts, links, rechts). Feste Steuerfolge aus 140 Befehlen, kein Sensor; **Verhalten** = Endposition. **Erfolg:** eine Endposition höchstens 2 Zellen vom Ziel.
- **Suche** (`ns_search.py`): derselbe Genetische Algorithmus wie in Stück 1 (100 Individuen, Turnier der Größe 3, die zwei Besten bleiben, Mutation Gauß σ = 0.25 auf 15 % der Gene, keine Kreuzung). Nur der Wert, nach dem ausgewählt wird, wechselt:
  - **Fitness:** −Abstand der Endposition zum Ziel (Bezug, mit denselben Seeds identisch zu Stück 1).
  - **Novelty:** mittlerer Abstand der Endposition zu den **k = 15 nächsten** Endpositionen der Vergleichsmenge aus Population (und Archiv). Der Punkt selbst zählt nicht, ein anderes Individuum am selben Ort zählt mit Abstand 0.
  - **Drei Archiv-Regeln:** *nur Population* (kein Archiv); *Archiv der Neuartigsten* (je Generation kommen die neuartigsten 5 % der Population, also 5 Punkte, ins Archiv); *Archiv mit Schwelle* (alle mit Novelty über ρ, höchstens 5 je Generation und
    dann die neuartigsten; bei Andrang ρ × 1.1, nach 3 Generationen ohne Neuzugang ρ × 0.9, Start ρ = 1).
- **Messgrößen:** Evaluationen bis zum ersten Erfolg; **besuchte Zellen** (Zellen, in denen mindestens eine Endposition lag); **wahrer Restweg** (kürzester Weg in Zellen von der besten besuchten Zelle zum Ziel, Breitensuche).
- **Gegenproben:** (1) Novelty von Hand auf Punkten einer Geraden (k = 1 und 2, Duplikate, Archivpunkt am selben Ort, k größer als die Vergleichsmenge); (2) **Orakel:** Novelty gegen SciPys `cKDTree` und gegen eine skalare Neuberechnung nur aus der
  Definition (mit und ohne Archiv, mehrere k, viele doppelte Punkte); (3) beide Archiv-Regeln von Hand (5 %, Schwelle, Andrang, Ruhe); (4) besuchte Zellen gegen eine Zählung über Zellen-Tupel, Restweg gegen SciPys Breitensuche; (5) das Ziel geht nicht in die
  Novelty-Auswahl ein (anderes Ziel, gleiche Novelty-Läufe); (6) Breitensuche, Kinematik und Lösbarkeit wie in Stück 1; (7) Studienläufe werden mit demselben Seed frisch reproduziert.
- **Vorgerechnete Studie** (`generate_precomputed.py` → `precomputed_sweep.json`, rund 14 Minuten parallel): je Stufe und Variante 20 Läufe à 400 Generationen (40 000 Evaluationen); ein Raster (k = 5 / 15 / 30 × 50 / 100 / 200 Individuen bei gleichem
  Gesamtbudget, Variante ohne Archiv) auf Stufe 3 und 4; und Stufe 4 mit dem Zehnfachen (4 000 Generationen, 400 000 Evaluationen) für Fitness, Novelty ohne Archiv und Archiv der Neuartigsten.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`. 20 Läufe je Zelle (Seeds 0 bis 19), 40 000 Evaluationen, sofern nicht anders genannt. Aufwand in Evaluationen (100 je Generation).

| Frage | Befund |
|---|---|
| Löst Novelty die Falle (Stufe 3)? | **Ja, in jedem Lauf.** Novelty ohne Archiv: **20 von 20**, Median 988.5 Evaluationen (schnellster Lauf 77, langsamster 3 373). Die Fitness: 11 von 20. Vorbehalt: der Fitness-Median (1 267) gilt nur für ihre 11 erfolgreichen Läufe, der von Novelty für alle 20. |
| Gilt das auf allen Stufen? | Erfolge 20 / 20 / 20 / **20** / **0** von 20 für Novelty auf Stufe 0 bis 4 (Fitness: 20 / 20 / 20 / 11 / 0). **Stufe 4 löst keine Variante.** |
| Ist Novelty schneller, wo die Fitness nicht täuscht? | **Nein.** Median 63.5 / 146 / 207.5 (Stufe 0 bis 2) gegen 63.5 / 126 / 190.5 für die Fitness; schlimmster Lauf auf Stufe 1 / 2: 307 / 447 gegen 252 / 875. Wer nicht täuscht, braucht Novelty nicht. |
| Warum löst Novelty die Falle? | Sie besucht **alle freien Zellen**: auf Stufe 0 bis 3 in jedem der 60 Läufe je Variante alle 484 / 465 / 465 / 465 Zellen; die Fitness im Mittel 327.15 / 376.95 / 422.45 / 343.9 (schlechtester Lauf auf Stufe 3: nur 191). Nach 50 Generationen auf Stufe 3 sind im Mittel 445.5 von 465 Zellen besucht (Fitness 272.6), nach 100 Generationen 463.15 (306.95). |
| Wie nah kommt die Fitness dem Ziel auf Stufe 3? | Der kleinste wahre Restweg liegt im Mittel bei 5.55 Zellen (schlechtester Lauf 17), bei Novelty bei 0. |
| Hilft ein Archiv? | **Nicht in diesem Labyrinth.** Beide Archiv-Varianten lösen Stufe 0 bis 3 in 20 von 20 Läufen, Median 63.5 / 146 / 207.5 / **1 195** (beide gleich) gegen 988.5 ohne Archiv; schlimmster Lauf auf Stufe 3: 4 006 (Neuartigste), 3 764 (Schwelle), 3 373 (ohne). Beide Regeln nehmen zu Beginn dieselben 5 Neuartigsten je Generation auf und unterscheiden sich erst, wenn die Schwelle greift. |
| Hängt es an k oder an der Population? | **Kaum.** Alle 9 Einstellungen (k = 5 / 15 / 30 × 50 / 100 / 200 Individuen) lösen Stufe 3 in 20 von 20 Läufen. Median 718.5 / 1 186.5 / 1 175 (k = 5), 659 / 988.5 / 1 308.5 (k = 15), 710 / 763.5 / 1 271.5 (k = 30): die Einstellung verschiebt den Aufwand, nicht den Erfolg. |
| Und auf Stufe 4? | **0 von 180 Läufen** in allen 9 Einstellungen. Der kleinste wahre Restweg liegt je Einstellung im Mittel zwischen 25.9 und 27.6 Zellen (Start: 77). |
| Wie weit kommt Stufe 4 mit 40 000 Evaluationen? | 0 / 0 / 0 / 0 von 20 (Fitness, Novelty, beide Archive). Besuchte Zellen im Mittel 223.8 / 264.15 / 257.45 / 255.35 von 427; kleinster Restweg 34.75 / 26.25 / 29.95 / 30.3 Zellen (bester Lauf 22 / 22 / 26 / 26). Novelty kommt dem Ziel also näher, erreicht es aber nie. |
| Hilft das Zehnfache an Budget? | **Nein.** Mit 400 000 Evaluationen 0 von 60 Läufen. Der mittlere Restweg fällt von 34.75 auf 32.3 (Fitness), von 26.25 auf 22.4 (Novelty ohne Archiv) und von 29.95 auf 23.75 (Archiv der Neuartigsten); der beste Lauf am Ende hat noch 20 / 16 / 21 Zellen. Novelty ohne Archiv: 26.25 (40 000), 24.75 (100 000), 23.85 (200 000), 22.4 (400 000); Fitness 34.75, 33.65, 32.95, 32.3: die Kurven flachen ab. Besuchte Zellen 238.8 / 288.2 / 286.4 von 427. |

## Befunde und Korrekturen gegenüber der Vorab-Messreihe

- **Die Vorab-Skizze nannte das Archiv „die 5 neuesten“; es sind die 5 neuartigsten** je Generation (die mit der größten Novelty). Das Schwellen-Archiv nimmt bei Andrang ebenfalls die neuartigsten 5, nicht die ersten 5 in Populationsreihenfolge
  wie die Skizze; deshalb liegt sein Median auf Stufe 3 jetzt bei 1 195 (gleich dem der anderen Archiv-Regel) statt bei 1 002.
- **Das Archiv war nicht nötig.** Erwartet war, dass das Archiv Novelty Search erst robust macht (Lehman und Stanley beschreiben es als festen Bestandteil). In diesem Labyrinth liefert die Variante ohne Archiv denselben Erfolg mit einer Vergleichsmenge von nur 100 statt (nach 400 Generationen) 2 100 Punkten: die Population
  allein bedeckt den kleinen Raum schnell genug. Das ist ein Befund über diese Aufgabe, nicht über Archive.
- **Novelty ist kein Allheilmittel für Stufe 4.** Die einfachste Hoffnung („breit genug suchen, dann findet sie den Weg“) trägt hier nicht: auch mit k, Population und zehnfachem Budget bleibt Stufe 4 ungelöst; die Suche bleibt im Mittel noch 22 bis 26 Zellen
  (Novelty ohne Archiv, 400 000 und 40 000 Evaluationen) vor dem Ziel hängen. Das ist der Anknüpfungspunkt für spätere Stücke der Linie.
- **Medianvergleich mit Vorbehalt.** Median 988.5 (Novelty, 20 von 20) gegen 1 267 (Fitness, nur die 11 erfolgreichen Läufe) ist kein Gleichstand der Stichproben; die Erfolgsquote ist der belastbare Vergleich.

## Ehrliche Grenzen

- **Das Verhaltensmaß ist von Hand gewählt** (die Endposition, zwei Zahlen). Das Ergebnis hängt an dieser Wahl; ein anderes Maß wird hier nicht untersucht.
- **Ein einziger Labyrinthtyp und eine feste Genomlänge.** Mit längerer Steuerfolge löst schon die Fitness Stufe 3 (Stück 1); die Vorteile von Novelty gelten für diese Parametrisierung.
- **20 Läufe je Zelle.** Erfolgsquoten wie 11 von 20 haben eine Standardabweichung von etwa 11 Prozentpunkten; kleine Unterschiede zwischen Archiv-Varianten und Einstellungen (etwa Median 988.5 gegen 1 195) liegen im Rauschen und werden nicht gedeutet.
- **Kein Archiv-Test in anderen Aufgaben.** Dass das Archiv hier nichts bringt, sagt nichts über größere oder offenere Verhaltensräume.
- **Das Schwellen-Archiv ist beim zehnfachen Budget nicht mitgerechnet** (das Archiv wächst dort auf Tausende Punkte; Rechenzeit).
- **Ein einfacher GA ohne Kreuzung;** Kreuzung und andere Operatoren würden die Zahlen ändern (nicht gemessen).

## Verwandte Demos im Portfolio

- [`deception-maze-demo`](https://github.com/sebastian-hanisch/deception-maze-demo): Stück 1, das täuschende Labyrinth und die zielgetriebene Suche.
- [`genetic-algorithm-demo`](https://github.com/sebastian-hanisch/genetic-algorithm-demo): der Genetische Algorithmus ohne Täuschung.
- [`nsga2-demo`](https://github.com/sebastian-hanisch/nsga2-demo): zwei Ziele statt Neuartigkeit.

## Bewusst nicht umgesetzt

Jede dieser Annahmen hebt ein Folgestück der Linie auf:

| Annahme | Folgestück |
|---|---|
| Das Verhaltensmaß ist richtig gewählt (hier: die Endposition) | Verhaltensraum (Folgestück) |
| Neuartigkeit allein genügt | Novelty plus Ziel (Folgestück) |
| Ein Archiv hilft | kein Folgestück |
| Eine einzige beste Lösung | MAP-Elites (Folgestück) |
| Die Suche braucht nur die Auswahl zu ändern | Go-Explore und POET (Folgestücke) |

## Tests

194 Tests, rund 80 Sekunden: Labyrinth und Roboter (aus Stück 1), Novelty von Hand und gegen `cKDTree` und eine skalare Definition, Archiv-Regeln, Läufe (Reproduzierbarkeit je Variante, Zählung der Evaluationen, besuchte Zellen, Restweg, das Ziel
geht nicht in die Auswahl ein), Auswertung und Vollständigkeit der vorgerechneten Datei (inklusive Neurechnung einzelner Studienläufe und eines verlängerten Laufs), Presets und Permalink, Diagramme (gesperrte Achsen), AppTest-Rauchtests mit festem
Würfel-Seed, der Smoke-Test der Portfolio-Vorlage, ein Quelltext-Test gegen Satz-Komma-Fehler und `test_claims.py` für jede Zahl dieser README.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `ns_maze.py`, `ns_robot.py` | Labyrinth, Breitensuche, Täuschungsmaße; Kinematik, Population, Pfad-Regler (wie Stück 1) |
| `ns_search.py` | Suche mit Fitness oder Novelty, Archiv-Regeln, besuchte Zellen, Restweg |
| `ns_evaluation.py` | Maße je Stufe, Lösbarkeit, Live-Paar, Zugriff auf die vorgerechnete Studie |
| `generate_precomputed.py` | rechnet die Studie vor → `precomputed_sweep.json` |
| `ns_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `ns_presets.py`, `ns_constants.py` | Presets, Permalink, Grenzen |
| `tests/` | siehe oben |

## Literatur

- Lehman, J., Stanley, K. O. (2011): Abandoning objectives: evolution through the search for novelty alone. *Evolutionary Computation* 19(2), 189–223 (Novelty als mittlerer Abstand zu den k nächsten Nachbarn, Archiv mit Schwelle).
- Lehman, J., Stanley, K. O. (2008): Exploiting open-endedness to solve problems through the search for novelty. *Proceedings of the Eleventh International Conference on Artificial Life (ALIFE XI)*, MIT Press.

## Lokal ausführen

```
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` und `python -m pytest tests/ -v`. Studie neu rechnen: `python generate_precomputed.py` (rund 14 Minuten).

Gebaut mit Streamlit und Plotly (die Simulation rechnet NumPy).
