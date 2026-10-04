"""Novelty Search – suchen, ohne das Ziel zu kennen - Stück 2 der Konzepte-Linie "Novelty Search und Quality-Diversity"
Sebastian Hanisch - Operations Research und Machine Learning

Derselbe Hof-Roboter und dasselbe täuschende Labyrinth wie in Stück 1 (deception-maze-demo), derselbe Genetische Algorithmus - nur wählt er nicht mehr nach dem Abstand zum Ziel aus, sondern nach Novelty: wie anders ist das
Verhalten eines Individuums (seine Endposition) als das der anderen? Siehe README für die Einordnung in die Linie.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import ns_constants as C
import ns_evaluation as E
import ns_maze as M
from ns_presets import (apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sync_query_params)
from ns_visualization import (build_budget_chart, build_coverage_chart, build_long_chart, build_progress_chart, build_run_figure, build_study_coverage_chart, build_success_chart, build_sweep_chart)

st.set_page_config(page_title="Novelty Search – Sebastian Hanisch", layout="wide")

PRE = E.load_precomputed()
N = PRE["seeds"]


@st.cache_data(show_spinner=False)
def _pair(level, gens, seed, variant, k):
    return E.live_pair(level, gens, seed, variant, k)


@st.cache_data(show_spinner=False)
def _solvability(level):
    return E.solvability(level)


def _evals(x):
    """Evaluationen mit Punkt als Tausendertrenner; ein Median zwischen zwei Läufen behält seine halbe Stelle (988.5, 1 308.5)."""
    if x is None:
        return "–"
    return C.fmt_int(int(x)) if x == int(x) else f"{x:,.1f}".replace(",", " ")


st.title("🧭 Novelty Search – suchen, ohne das Ziel zu kennen")
st.markdown(
    """
Im ersten Stück lief ein Roboter in die **Sackgasse**: die Fitness „möglichst nah am Ziel“ belohnte jede Fahrt, die an der Wand unter dem Ziel endete. **Novelty Search** dreht die Frage um: Ausgewählt wird nicht, wer dem Ziel am nächsten
endet, sondern wer **anders endet als alle anderen**. Das Ziel kommt in der Auswahl gar nicht mehr vor. Weil die Suche dadurch den ganzen Raum ausbreitet, läuft sie früher oder später auch durch die Öffnung in der Wand. Dieses Stück
zeigt, **wann das gelingt, was ein Archiv der bisherigen Verhalten dazu beiträgt** (in diesem Labyrinth: nichts) und **wo Novelty Search scheitert** (Stufe 4).
"""
)
st.caption(
    "Stück 2 der Linie „Novelty Search und Quality-Diversity“; Labyrinth, Roboter und Genetischer Algorithmus wie in [Stück 1](https://sebastianhanisch-deception-maze-demo.streamlit.app/). "
    "Das Verhalten ist hier von Hand gewählt (die Endposition); das nächste Stück fragt, was passiert, wenn man es anders wählt."
)

with st.expander("So funktioniert die Novelty-Auswahl", expanded=True):
    st.markdown(
        """
- **Verhalten:** die Endposition der Fahrt (zwei Zahlen). Zwei Individuen sind „verschieden“, wenn ihre Endpositionen weit auseinanderliegen, egal wie ähnlich die Steuerfolgen sind.
- **Novelty** eines Individuums: der **mittlere Abstand zu den k nächsten** Endpositionen in der Vergleichsmenge (Population, bei den Archiv-Varianten zusätzlich das Archiv). Wer in einer dicht besetzten Gegend endet, hat wenig Novelty; wer allein in
  einer leeren Ecke endet, viel. Ausgewählt wird nach Novelty (Turnier, Elite, Mutation wie bei der Fitness).
- **Archiv-Regeln:** *nur Population*: kein Gedächtnis, nur die aktuelle Generation. *Archiv der Neuartigsten*: je Generation kommen die neuartigsten 5 % ins Archiv. *Archiv mit Schwelle*: alle über der Schwelle ρ (höchstens 5 je Generation); ρ
  steigt bei Andrang und fällt, wenn 3 Generationen lang keiner die Schwelle erreicht.
- **Das Ziel** dient nur der Messung: **Erfolg**, sobald eine Endposition höchstens 2 Zellen vom Ziel liegt. Der **wahre Restweg** ist der kürzeste Weg in Zellen von der besten bisher besuchten Zelle zum Ziel (Breitensuche).
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_cols = st.columns(len(C.PRESET_ORDER))
for i, name in enumerate(C.PRESET_ORDER):
    with preset_cols[i]:
        st.button(name, key=f"preset_{name}", width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name] or None)

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    level = st.select_slider("Labyrinth (Stufe)", options=C.LEVELS, key="level_select", format_func=lambda v: C.LEVEL_LABELS[v], help="0 freies Feld bis 4 Schlangenlinie; je höher, desto länger der Umweg.")
    variant = st.selectbox("Novelty-Variante", options=C.NOVELTY_VARIANTS, key="variant_select", format_func=lambda v: C.VARIANT_LABELS[v],
                           help="Welche Archiv-Regel die Novelty-Suche im Live-Lauf benutzt. Die Fitness-Suche läuft immer daneben (gleicher Seed, gleiche Anfangspopulation).")
    k = st.slider("Nachbarn k für die Novelty", *bounds("k_slider"), step=C.K_STEP, key="k_slider", help="Die Novelty ist der mittlere Abstand zu den k nächsten Endpositionen der Vergleichsmenge.")
    gens = st.slider("Generationen im Live-Lauf", *bounds("gens_slider"), step=C.GENS_STEP, key="gens_slider", help=f"Je Generation {C.POP} Individuen, also {C.POP} Evaluationen.")
    seed = st.number_input("Zufalls-Seed", min_value=bounds("seed_input")[0], max_value=bounds("seed_input")[1], step=1, key="seed_input", help="Bestimmt die Anfangspopulation und alle Zufallsentscheidungen; beide Suchen starten gleich.")
    st.button("🎲 Neuen Lauf würfeln", on_click=randomize_seed)

level, variant, k, gens, seed = int(level), str(variant), int(k), int(gens), int(seed)
sync_query_params({"level_select": level, "variant_select": variant, "k_slider": k, "gens_slider": gens, "seed_input": seed})

grid = M.make_maze(level)
lm = E.level_metrics(level)
free = lm["free"]

st.markdown("---")
st.markdown("## 🔗 Wie täuschend ist diese Stufe?")
st.caption(f"{C.LEVEL_LABELS[level]}. Die Maße stammen aus Stück 1: sie sagen, wie sehr die Fitness „Abstand zum Ziel“ in die Irre führt; Novelty benutzt sie nicht.")
m1 = st.columns(4)
m1[0].metric("Kürzester Weg", f"{lm['path']} Zellen", delta=f"{lm['detour']:.2f}-facher Umweg", delta_color="off")
m1[1].metric("Fitness-Distanz-Korrelation", f"{lm['fdc']:.2f}", help="1 = die Luftlinie führt zum Ziel; je kleiner, desto täuschender.")
m1[2].metric("Fallen (lokale Minima)", f"{lm['minima']}", help="Zellen, in denen die Luftlinie zum Ziel kleiner ist als in allen freien Nachbarzellen.")
m1[3].metric("Freie Zellen", f"{free}", help="So viele Zellen kann eine Endposition belegen; die Novelty-Suche versucht, alle zu besuchen.")
dist_ok, steps_used = _solvability(level)
st.caption(f"Lösbar? Ja: eine aus dem kürzesten Pfad abgeleitete Steuerfolge fährt ins Ziel (Abstand am Ende {dist_ok:.2f} Zellen, {steps_used} von {C.STEPS} Schritten benutzt).")

st.markdown("---")
st.subheader("📐 Ein Lauf: Fitness gegen Novelty")
with st.spinner(f"Lasse beide Suchen {gens} Generationen laufen …"):
    (fit, fit_track), (nov, nov_track) = _pair(level, gens, seed, variant, k)
fit_ok, nov_ok = fit.first_solved is not None, nov.first_solved is not None
c_fit, c_nov = st.columns(2)
with c_fit:
    st.markdown(f"**{C.VARIANT_LABELS['fitness']}**")
    f1 = st.columns(2)
    f1[0].metric("Fitness: Ergebnis", "Ziel erreicht" if fit_ok else "Ziel verfehlt")
    f1[1].metric("Fitness: Evaluationen bis zum Erfolg", _evals(fit.first_solved))
    f2 = st.columns(2)
    f2[0].metric("Fitness: besuchte Zellen", f"{fit.visited_cells[-1]} von {free}")
    f2[1].metric("Fitness: kleinster Restweg", f"{fit.best_progress[-1]:.0f} Zellen")
    st.plotly_chart(build_run_figure(grid, fit.end_positions[-1], fit_track, fit.visited_grid), width="stretch", key=f"run_fit_{level}_{gens}_{seed}")
with c_nov:
    st.markdown(f"**{C.VARIANT_LABELS[variant]}** (k = {k})")
    n1 = st.columns(2)
    n1[0].metric("Novelty: Ergebnis", "Ziel erreicht" if nov_ok else "Ziel verfehlt")
    n1[1].metric("Novelty: Evaluationen bis zum Erfolg", _evals(nov.first_solved))
    n2 = st.columns(2)
    n2[0].metric("Novelty: besuchte Zellen", f"{nov.visited_cells[-1]} von {free}")
    n2[1].metric("Novelty: kleinster Restweg", f"{nov.best_progress[-1]:.0f} Zellen")
    st.plotly_chart(build_run_figure(grid, nov.end_positions[-1], nov_track, nov.visited_grid, nov.archive), width="stretch", key=f"run_nov_{level}_{gens}_{seed}_{variant}_{k}")
st.caption("Hellblau: alle Zellen, in denen eine Fahrt je endete. Blaue Punkte: die Endpositionen der letzten Generation. Orange: die beste Fahrt des Laufs. Violett (nur mit Archiv): die gespeicherten Endpositionen.")
col_c, col_d = st.columns(2)
with col_c:
    st.markdown("**Bester Abstand zum Ziel**")
    st.plotly_chart(build_progress_chart(fit.best_distance, nov.best_distance, variant, lm["trap"]), width="stretch", key=f"progress_{level}_{gens}_{seed}_{variant}_{k}")
with col_d:
    st.markdown("**Besuchte Zellen**")
    st.plotly_chart(build_coverage_chart(fit.visited_cells, nov.visited_cells, variant, free), width="stretch", key=f"coverage_{level}_{gens}_{seed}_{variant}_{k}")
if nov_ok and not fit_ok:
    st.info(f"Die Novelty-Suche erreicht das Ziel nach {C.fmt_int(nov.first_solved)} Evaluationen, die Fitness-Suche nicht (bester Abstand {min(fit.best_distance):.2f} Zellen). Das ist ein Lauf; ob es typisch ist, zeigt die Studie unten.")
elif fit_ok and not nov_ok:
    st.info(f"Hier erreicht die Fitness-Suche das Ziel (nach {C.fmt_int(fit.first_solved)} Evaluationen), die Novelty-Suche in {gens} Generationen noch nicht (bester Abstand {min(nov.best_distance):.2f} Zellen). Novelty ist kein Allheilmittel: "
            "wo die Fitness nicht täuscht, ist sie oft gleich schnell oder schneller.")
elif fit_ok and nov_ok:
    st.info(f"Beide Suchen erreichen das Ziel: die Fitness nach {C.fmt_int(fit.first_solved)}, die Novelty nach {C.fmt_int(nov.first_solved)} Evaluationen. Wo die Fitness nicht täuscht, hat Novelty keinen Vorteil.")
else:
    st.info(f"Keine der beiden Suchen erreicht das Ziel in {gens} Generationen (bester Abstand Fitness {min(fit.best_distance):.2f}, Novelty {min(nov.best_distance):.2f} Zellen; kleinster Restweg {fit.best_progress[-1]:.0f} gegen "
            f"{nov.best_progress[-1]:.0f} Zellen). Mehr Generationen oder ein anderer Seed können das ändern; die Studie unten fasst {N} Läufe zusammen.")

st.markdown("---")
st.subheader("🔬 Wie oft gelingt es? Die Studie über 20 Läufe")
st.markdown(
    f"Je Stufe und Variante {N} Läufe mit unterschiedlichem Seed, jeweils bis zu {C.fmt_int(PRE['budget'])} Evaluationen ({C.STUDY_GENS} Generationen). **Erfolg** heißt: irgendein Individuum endet höchstens {C.SUCCESS_RADIUS:g} Zellen vom Ziel."
)
st.plotly_chart(build_success_chart(PRE), width="stretch", key="success_chart")
study_level = st.select_slider("Stufe für Tabelle und Verläufe", options=list(C.LEVELS), value=C.DEFAULT_LEVEL, key="study_level_select", format_func=lambda v: C.LEVEL_LABELS[v])
free_s = E.level_info(PRE, study_level)["free"]
rows = []
for v in C.VARIANTS:
    c = E.study_cell(PRE, study_level, v)
    rows.append(f"| {C.VARIANT_LABELS[v]} | {E.success_count(c['runs'])} von {N} | {_evals(E.median_solved(c['runs']))} | {sum(c['cells']) / N:.2f} von {free_s} | {sum(c['progress']) / N:.2f} (bester Lauf {min(c['progress']):.0f}) |")
st.markdown("| Variante | Erfolg | Median der Evaluationen bis zum Erfolg | besuchte Zellen (Mittel) | kleinster wahrer Restweg am Ende (Mittel) |\n|---|---|---|---|---|\n" + "\n".join(rows))
col_e, col_f = st.columns(2)
with col_e:
    st.markdown("**Erfolgsquote über dem Aufwand**")
    st.plotly_chart(build_budget_chart(PRE, study_level), width="stretch", key=f"budget_chart_{study_level}")
with col_f:
    st.markdown("**Besuchte Zellen über den Generationen**")
    st.plotly_chart(build_study_coverage_chart(PRE, study_level), width="stretch", key=f"study_coverage_{study_level}")
fit3, nov3, fit4 = E.study_cell(PRE, 3, "fitness"), E.study_cell(PRE, 3, "novelty"), E.study_cell(PRE, 4, "fitness")
easy = [E.median_solved(E.study_cell(PRE, lv, "fitness")["runs"]) for lv in (0, 1, 2)]
easy_n = [E.median_solved(E.study_cell(PRE, lv, "novelty")["runs"]) for lv in (0, 1, 2)]
st.info(
    f"**Stufe 3 (die Falle):** Novelty ohne Archiv löst {E.success_count(nov3['runs'])} von {N} Läufen (Median {_evals(E.median_solved(nov3['runs']))} Evaluationen), die Fitness {E.success_count(fit3['runs'])} von {N} "
    f"(Median {_evals(E.median_solved(fit3['runs']))}, aber nur über deren erfolgreiche Läufe). Novelty besucht dabei alle {E.level_info(PRE, 3)['free']} freien Zellen, die Fitness im Mittel {sum(fit3['cells']) / N:.0f}. "
    f"**Stufen 0 bis 2:** kein Vorteil (Stufe 0 gleich, Stufe 1 und 2 langsamer): Median {_evals(easy_n[0])} / {_evals(easy_n[1])} / {_evals(easy_n[2])} Evaluationen gegen {_evals(easy[0])} / {_evals(easy[1])} / {_evals(easy[2])} für die Fitness. "
    f"**Das Archiv** ändert an der Erfolgsquote nichts: beide Varianten mit Archiv lösen Stufe 3 in {E.success_count(E.study_cell(PRE, 3, 'archive_top')['runs'])} von {N} Läufen "
    f"(Schwelle: {E.success_count(E.study_cell(PRE, 3, 'archive_threshold')['runs'])}) und sind nicht schneller als die Variante ohne Archiv (Median {_evals(E.median_solved(E.study_cell(PRE, 3, 'archive_top')['runs']))} gegen "
    f"{_evals(E.median_solved(nov3['runs']))}). **Stufe 4** löst keine Variante ({E.success_count(fit4['runs'])} von {N} bei der Fitness, "
    f"{E.success_count(E.study_cell(PRE, 4, 'novelty')['runs'])} bei Novelty)."
)

st.markdown("---")
st.subheader("🔬 Wie viel hängt an k und der Population?")
sweep_level = st.select_slider("Stufe für das Raster", options=list(C.SWEEP_LEVELS), value=C.SWEEP_LEVELS[0], key="sweep_level_select", format_func=lambda v: C.LEVEL_LABELS[v])
st.markdown(
    f"Variante „Novelty, nur Population“, gleiches Gesamtbudget ({C.fmt_int(C.STUDY_BUDGET)} Evaluationen), aber andere Einstellungen: Zahl der Nachbarn k = 5 / 15 / 30 und Populationsgröße 50 / 100 / 200 "
    f"(entsprechend 800 / 400 / 200 Generationen). Je {N} Läufe."
)
st.plotly_chart(build_sweep_chart(PRE, sweep_level), width="stretch", key=f"sweep_{sweep_level}")
sw = lambda lv, k_, p: E.success_count(E.sweep_cell(PRE, lv, k_, p)["runs"])
cells3 = [sw(3, k_, p) for k_ in C.SWEEP_KS for p in C.SWEEP_POPS]
total4 = sum(E.success_count(x["runs"]) for x in PRE["sweep"] if x["level"] == 4)
prog4 = [sum(x["progress"]) / N for x in PRE["sweep"] if x["level"] == 4]
st.info(
    f"Auf Stufe 3 lösen alle neun Einstellungen {min(cells3) if min(cells3) == max(cells3) else str(min(cells3)) + ' bis ' + str(max(cells3))} von {N} Läufen (Standard k = 15, 100 Individuen: {sw(3, 15, 100)}); die Einstellung verschiebt nur, "
    f"wie schnell: Median {_evals(min(E.median_solved(E.sweep_cell(PRE, 3, k_, p)['runs']) for k_ in C.SWEEP_KS for p in C.SWEEP_POPS))} bis {_evals(max(E.median_solved(E.sweep_cell(PRE, 3, k_, p)['runs']) for k_ in C.SWEEP_KS for p in C.SWEEP_POPS))} Evaluationen. Auf Stufe 4 gelingt es in **keiner** "
    f"({total4} Erfolge in {len(C.SWEEP_KS) * len(C.SWEEP_POPS) * N} Läufen); der kleinste wahre Restweg liegt je Einstellung im Mittel zwischen {min(prog4):.1f} und {max(prog4):.1f} Zellen "
    f"(Start: {E.level_info(PRE, 4)['path']})."
)

st.markdown("---")
st.subheader("🔬 Hilft mehr Budget auf Stufe 4?")
long_budget = PRE["long"]["budget"]
st.markdown(
    f"Stufe 4 mit dem **Zehnfachen** des Budgets ({C.fmt_int(long_budget)} statt {C.fmt_int(PRE['budget'])} Evaluationen, {C.LONG_GENS} Generationen), je {N} Läufe. Die Abbildung zeigt, wie nah dem Ziel (entlang des wahren Wegs) die Suche im Mittel kommt."
)
st.plotly_chart(build_long_chart(PRE), width="stretch", key="long_chart")
idx40 = PRE["budget"] // (C.LONG_STEP * C.POP) - 1
rows = []
for v in C.LONG_VARIANTS:
    c = E.long_cell(PRE, v)
    rows.append(f"| {C.VARIANT_LABELS[v]} | {E.success_count(c['runs'])} von {N} | {c['mean_progress'][idx40]:.2f} | {sum(c['progress']) / N:.2f} (bester Lauf {min(c['progress']):.0f}) | {sum(c['cells']) / N:.2f} von {E.level_info(PRE, 4)['free']} |")
st.markdown(f"| Variante | Erfolg nach {C.fmt_int(long_budget)} | Restweg nach {C.fmt_int(PRE['budget'])} (Mittel) | Restweg nach {C.fmt_int(long_budget)} (Mittel) | besuchte Zellen (Mittel) |\n|---|---|---|---|---|\n" + "\n".join(rows))
lf, ln = E.long_cell(PRE, "fitness"), E.long_cell(PRE, "novelty")
st.info(
    f"Auch mit {C.fmt_int(long_budget)} Evaluationen erreicht **keine** der drei Varianten das Ziel ({E.success_count(lf['runs']) + E.success_count(ln['runs']) + E.success_count(E.long_cell(PRE, 'archive_top')['runs'])} Erfolge in {3 * N} Läufen). "
    f"Novelty kommt dem Ziel im Mittel näher als die Fitness (Restweg {sum(ln['progress']) / N:.2f} gegen {sum(lf['progress']) / N:.2f} Zellen), bleibt aber weit davon entfernt, und die Kurve flacht ab: "
    "Mehr Rechenzeit löst das Problem nicht, die Suche braucht einen anderen Ansatz."
)

st.markdown("---")
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Das Verhaltensmaß ist richtig gewählt** (hier: die Endposition) | Novelty misst nur Unterschiede im gewählten Maß. Ein Maß, das das Wesentliche nicht trennt (zwei Fahrten mit gleicher Endposition, aber verschiedenem Weg), macht die Suche blind dafür. | **Verhaltensraum** (Folgestück): andere Maße, ihr Einfluss auf die Suche |
| **Neuartigkeit allein genügt** | Ohne Ziel sucht die Suche überall gleich gern, auch dort, wo es nicht zum Ziel führt; je größer der Raum, desto dünner die Abdeckung (Stufe 4: nicht gelöst). | **Novelty plus Ziel** (Folgestück): beides zusammen in einer Auswahl |
| **Ein Archiv hilft** | In diesem Labyrinth hilft es nicht: dieselbe Erfolgsquote bei einer Vergleichsmenge, die bei 400 Generationen von 100 auf 2 100 Punkte wächst. Ob ein Archiv in anderen Aufgaben nötig wird, ist hier nicht gemessen. | kein Folgestück |
| **Eine einzige beste Lösung** | Die Suche liefert den Weg zum Ziel, nicht die Menge der verschiedenen guten Wege. | **MAP-Elites** (Folgestück): Archiv nach Verhaltensbereichen |
| **Die Suche braucht nur die Auswahl zu ändern** | Stufe 4 bleibt auch mit zehnfachem Budget ungelöst: es fehlt ein Mechanismus, der erreichte Zwischenstände gezielt weiter ausbaut. | **Go-Explore** und **POET** (Folgestücke) |
"""
)
st.caption(
    "Verwandt im Portfolio: [deception-maze-demo](https://sebastianhanisch-deception-maze-demo.streamlit.app/) (Stück 1: das täuschende Labyrinth), "
    "[genetic-algorithm-demo](https://sebastianhanisch-genetic-algorithm-demo.streamlit.app/) (der Genetische Algorithmus), [nsga2-demo](https://sebastianhanisch-nsga2-demo.streamlit.app/) (zwei Ziele)."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Verhalten.** $b(g)\in\mathbb{R}^2$ ist die Endposition der Fahrt mit der Steuerfolge $g$ (Kinematik wie in Stück 1).

**Novelty.** Für ein Individuum $i$ der Population $P$ mit Archiv $A$ und Vergleichsmenge $R = P \cup A$:
$\rho_k(i) = \frac{1}{k}\sum_{j \in N_k(i)} \lVert b(g_i) - b(g_j) \rVert_2$, wobei $N_k(i)$ die $k$ Elemente von $R\setminus\{i\}$ mit dem kleinsten Abstand sind (ein anderes Individuum am selben Ort zählt mit Abstand 0;
bei weniger als $k$ anderen Elementen gehen alle ein). Ausgewählt wird nach $\rho_k$ (Turnier der Größe 3, Elite 2) statt nach $f(g)=-\lVert b(g)-z\rVert_2$.

**Archiv.** *Neuartigste:* je Generation die $\lceil 0.05\,|P|\rceil$ Individuen mit größtem $\rho_k$. *Schwelle:* alle mit $\rho_k > \tau$, höchstens 5 (die neuartigsten); bei mehr Kandidaten $\tau \leftarrow 1.1\,\tau$, nach 3 Generationen
ohne Neuzugang $\tau \leftarrow 0.9\,\tau$; Start $\tau = 1$.

**Messgrößen.** Erfolg: $\lVert b(g)-z\rVert_2 < 2$. Besuchte Zellen: Zahl der Zellen, die mindestens eine Endposition enthielten. Wahrer Restweg: $\min d(c)$ über die besuchten Zellen $c$, $d$ = Breitensuche zum Ziel.

Implementiert in `ns_search.py` (Novelty, Archiv, Lauf), `ns_maze.py` und `ns_robot.py` (wie Stück 1).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html))."
)
