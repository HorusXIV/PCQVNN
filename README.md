# Quaternion-Valued Neural Networks and Predictive Coding

## Projektidee

In diesem Projekt möchte ich untersuchen, ob ein quaternionwertiges neuronales Netzwerk (QVNN) mit Predictive Coding (PC) vernünftig trainiert werden kann und wie sich ein solches Modell bei Standardaufgaben zur Approximation nichtlinearer Funktionen im Vergleich zu einem starken Quaternion-Backpropagation-Basismodell schlägt.  

Konkret besteht das Ziel darin, die Benchmark-Einstellung von Bill, Cox und Champagne (Bill et al., 2023) zu erweitern, indem die Quaternion-Backpropagation durch ein auf Predictive Coding basierendes Trainingsverfahren ersetzt und beide Ansätze unter derselben Architektur und denselben Benchmark-Funktionen verglichen werden. 

## Quaternion-Valued Neural Networks (QVNN)

Ein quaternionwertiges neuronales Netzwerk (QVNN) ist ein neuronales Netzwerk, in dem Eingaben, Gewichte und Aktivierungen nicht als reelle Zahlen, sondern als **Quaternionen** dargestellt werden. Ein Quaternion hat vier Komponenten und kann als
$$
q = a + b\mathbf{i} + c\mathbf{j} + d\mathbf{k}
$$
geschrieben werden. Im Gegensatz zu reellen Zahlen ist die Quaternionenmultiplikation über das Hamilton-Produkt nicht kommutativ, also gilt im Allgemeinen $q_1 \otimes q_2 \neq q_2 \otimes q_1$. Deswegen sind QVNNs schwieriger zu trainieren als gewöhnliche reellwertige Netze, bieten allerdings auch einige Vorteile. 

Dadurch dass, das Neuron durch vier miteinander Verknüpften Werte besteht, bietet ein QVNN besonders dann Vorteile, wenn Daten aus vier zusammenhängenden Komponenten bestehen (z.B. RGBA, 3D Rotationen, Signalverarbeitung). Die Literatur zeigt zudem, dass Quaternion-Netzwerke in verschiedenen Anwendungen kompaktere Modelle und teils bessere Generalisierung als vergleichbare reellwertige Netze liefern können (Takahashi et al., 2025). Für Quaternion-Netze existiert jedoch keine einheitliche Standard-Backpropagation; stattdessen gibt es mehrere Varianten wie Arena, Nitta, CMLP und GHR (Bill et al., 2023).  

## Predictive Coding (PC)

Predictive Coding ist ein alternatives Lernprinzip zu klassischer Backpropagation. Die Grundidee ist, dass ein Modell nicht  nur vorwärts rechnet, sondern auf jeder Ebene versucht, die darunterliegende Aktivität vorherzusagen. Gelernt wird dann über die Minimierung von Vorhersagefehlern. Dabei werden zunächst latente Zustände iterativ angepasst (Inferenz) und anschliessend die Gewichte aktualisiert (lernen). Das führt zu einer charakteristischen Dynamik aus schneller Inferenz und langsamerem Gewichtslernen (Stenlund, 2025). 

Ein wesentlicher Vorteil von PC ist, dass die Gewichtsupdates lokal formuliert sind: Sie hängen nur von Aktivität und Fehlern benachbarter Schichten ab. Dadurch ist PC konzeptionell näher an biologisch inspirierten Lernmechanismen und zugleich interessant für dezentrale oder parallele Rechenstrukturen. Ausserdem erlaubt PC prinzipiell adaptive Inferenz, bei der einfache Beispiele mit wenigen Schritten auskommen, während schwierigere Beispiele mehr Iterationen benötigen (Stenlund, 2025).  

## Projektziel & methodik

Das Ziel dieses Projekt soll eine Erweiterung der Vergleichsstudie von Bill et al. (2023) um ein Modell, welche auf Predictive Coding (PC) basiert. Das Hauptziel ist die Untersuchung, ob PC-basierte Optimierung eine valide Alternative zur klassischen Backpropagation in der Quaternionenalgebra $\mathbb{H}$ darstellt.
### Kernherausforderungen in $\mathbb{H}$
Aufgrund der Isomorphie zu $\mathbb{R}^4$ und der spezifischen algebraischen Struktur von $\mathbb{H}$ sind viele Standard-Aktivierungsfunktionen (z. B. ReLU) im Sinne des Hamilton-Kalküls nicht holomorph. Dies führt bei herkömmlicher Backpropagation zu:

* Extrem komplexen Kettenregeln und Produktverkettungen.
* Hohem Rechenaufwand im GHR-Kalkül.

### Der Predictive Coding Ansatz
Durch den Einsatz von PC wird die globale Fehlerfortpflanzung durch lokale Fehlerminimierung zwischen den Schichten ersetzt. Dies bietet folgende Vorteile:

   1. Vermeidung langer Ableitungsketten: Die Optimierung erfolgt schichtweise, was die mathematische Komplexität der Gradientenberechnung reduziert.
   2. Parallelisierungspotenzial: Lokale Rechenoperationen ermöglichen effizientere Hardware-Implementierungen.
   3. Architektur: Mit einer 1-5-5-5-1 Struktur bleibt das Netzwerk flach genug, um die bekannten Konvergenzprobleme von PC in tiefen Architekturen zu vermeiden.

## Praktische Trainingsstrategie

Das Projekt behandelt PC und QVNN nicht nur als mathematische Kombination, sondern als Rechenproblem. Predictive Coding ersetzt einen einzelnen Backpropagation-Schritt durch eine innere Inferenzschleife über latente Zustände. Quaternionwertige Schichten erhöhen zusätzlich die Kosten jeder linearen Operation, weil ein Hamilton-Produkt vier gekoppelte Real-Komponenten verarbeitet. Die zentrale Frage ist deshalb nicht nur, ob das Verfahren konvergiert, sondern ob es mit vertretbarer Trainingszeit skaliert.

Für die PCN-Experimente werden die Zielwerte auf Basis der Trainingsmenge normalisiert. Die Benchmark-Funktionen haben sehr unterschiedliche Skalen, insbesondere Schwefel. Ohne Zielnormalisierung dominiert die Zielskala die latente Inferenz und die Readout-Gewichte können instabil werden. Trainiert wird daher intern auf standardisierten Skalarzielen; die Zusammenfassungen und Testmetriken werden auf der ursprünglichen Zielskala berichtet.

Die latente Inferenz verwendet eine feste maximale Schrittzahl, kann aber adaptiv früher abbrechen, wenn die mittlere Norm der latenten Gradienten klein genug ist. Das folgt der PC-Idee, dass einfache Beispiele weniger Inferenz benötigen als schwierige Beispiele. Dadurch sinkt die Trainingszeit, ohne die maximale Inferenzkapazität für harte Batches zu entfernen.

Im GPU-Betrieb werden teure Finite-Checks nicht in jedem Inferenzschritt erzwungen. Diese Checks sind hilfreich zur Diagnose, synchronisieren aber CPU und GPU und verlangsamen lange H200-Läufe. Für Debugging können sie wieder aktiviert werden; für Produktionsläufe ist die stabilere Kombination aus Zielnormalisierung, kleinen Lernraten und adaptiver Inferenz sinnvoller.

## Hybride PC/QVNN-Architektur

Das erste skalierbare Hybridmodell ist ein split-quaternion Predictive Coding Network. Es verwendet quaternionwertige Zustände, Gewichte, lokale Fehler und Hamilton-Produkte, aber komponentenweise Split-Aktivierungen. Dieser Ansatz ist bewusst pragmatisch: Er erhält die Kopplung der vier Quaternion-Komponenten, vermeidet aber die deutlich teureren GHR-Ableitungsterme in jeder latenten Inferenziteration.

Die GHR-Variante bleibt als mathematisch reichere Forschungsrichtung interessant, sollte aber erst nach einer stabilen split-quaternion PCN-Basis verfolgt werden. In einer PCN-Schleife würden GHR-Terme nicht nur einmal pro Backward-Pass, sondern wiederholt pro Inferenzschritt anfallen. Das macht GHR-PCN zum Genauigkeits-/Fidelity-Experiment, nicht zum ersten skalierbaren Default.

Quaternion-Matrixmultiplikation wird für den üblichen Layer-Fall über gekoppelte Real-GEMMs formuliert. Das vermeidet grosse temporäre Tensoren der Form `(batch, out, in, 4)` und nutzt die optimierten Matrixmultiplikationen von PyTorch besser aus. Für kleine Forschungsmodelle ist der Unterschied moderat; für breitere PC/QVNN-Varianten wird dieser Punkt wichtig.

### Baseline & Evaluation
Als primäre Baseline dient das CMLP (Clifford Multilayer Perceptron)-Verfahren, das in der Vergleichsstudie von Bill et al. die besten Resultate lieferte. Evaluiert wird die Performance bei der Approximation nichtlinearer Funktionen hinsichtlich:

* Approximationsgenauigkeit (MSE)
* Konvergenzgeschwindigkeit
* Recheneffizienz

## Experimente auf dem H200

Die PCN-Benchmarks sind für H200-Läufe so ausgelegt, dass Daten einmal auf das Zielgerät verschoben werden, die Trainingsziele normalisiert sind und die Validierung nicht in jedem Epochenschritt eine volle teure Inferenz erzwingt. Für einen vollständigen PCN-Benchmark über die vier Funktionen:

```bash
sbatch cluster/run_pcn_benchmark.slurm
```

Für die Hyperparametersuche:

```bash
sbatch cluster/run_pcn_hyperparameter_search.slurm
```

Für das split-quaternion Predictive Coding Network alleine:

```bash
sbatch cluster/run_cpcn_benchmark.slurm
```

Dieser CPCN-Lauf ist der direkte Test der CMLP-inspirierten PC/QVNN-Idee: Die Daten bleiben quaternionwertig, die Zielwerte werden für die PC-Inferenz stabilisiert, und die Auswertung wird wieder auf der ursprünglichen Skala berichtet.

Die wichtigsten PCN-Parameter sind:

* `T_infer_train`: maximale Anzahl latenter Inferenzschritte während des Trainings.
* `T_infer_eval`: maximale Anzahl latenter Inferenzschritte während der Auswertung.
* `eta_infer`: Schrittweite der latenten Zustandsinferenz.
* `eta_learn`: Schrittweite der Gewichtsaktualisierung.
* `batch_size`: bestimmt den Kompromiss zwischen verrauschten lokalen Updates und GPU-Auslastung.
* `adaptive_inference`: erlaubt frühen Abbruch der Inferenz, wenn die latenten Gradienten klein genug sind.
* `normalize_targets`: stabilisiert Funktionen mit stark unterschiedlichen Zielskalen und sollte für PCN-Läufe aktiviert bleiben.

## Repostruktur

```text
configs/     Hyperparameter fuer CMLP/GHR-Vergleiche
cluster/     Apptainer-Definition und SLURM-Skripte fuer HPC-Laeufe
notebooks/   Explorative Checks und Analyse-Notebooks
results/     Gespeicherte Experimentresultate und Modellartefakte
src/         Python-Package `qvnn_pc`
tests/       Pytest-Suite fuer Algebra, Datenpipeline, Modelle und Benchmarks
report/      LaTeX-Abgabebericht
```

Die wichtigsten Einstiegspunkte im Code sind:

* `qvnn_pc.data`: Benchmark-Datengenerierung und Train/Val/Test-Splits.
* `qvnn_pc.models`: CMLP, GHR, klassisches MLP, real-valued PCN und split-quaternion PCN.
* `qvnn_pc.architecture_comparison`: CMLP/GHR-Vergleich.
* `qvnn_pc.pcn_benchmark`, `qvnn_pc.cpcn_benchmark`: PCN- und CPCN-Laeufe.
* `qvnn_pc.pcn_hyperparameter_search`: Hyperparametersuche fuer das real-valued PCN.

## Setup

Mit `uv`:

```bash
uv sync
uv run pytest tests -q
uv run python -m qvnn_pc.architecture_comparison --benchmark ackley --output results/local_archcmp_ackley.csv
```

Ohne `uv`, mit einer klassischen virtuellen Umgebung:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pytest tests -q
```

Die Cluster-Laeufe verwenden das Container-Setup aus `cluster/pcqvnn.def`. Das Image wird einmal gebaut und danach mit den SLURM-Skripten gestartet:

```bash
apptainer build cluster/pcqvnn.sif cluster/pcqvnn.def
sbatch cluster/run_architecture_comparison.slurm
```

## References

*   Bill, J., Champagne, L., Cox, B., & Bihl, T. (2021). [Meta-Heuristic Optimization Methods for Quaternion-Valued Neural Networks](https://doi.org). *Mathematics*, 9(938).
*   Bill, J., Cox, B. A., & Champagne, L. (2023). [A Comparison of Quaternion Neural Network Backpropagation Algorithms](https://doi.org). *Expert Systems with Applications*, 232, 120448.
*   Stenlund, M. (2025). [Introduction to Predictive Coding Networks for Machine Learning](https://arxiv.org/abs/2506.06332). *arXiv preprint arXiv:2506.06332*.
*   Takahashi, K., Fujimoto, A., Kawakami, S., Mitera, H., & Hashimoto, M. (2025). [Optimal Predictive Control Using Quaternion-Valued Neural Network-Based System Identification](https://doi.org). *Journal of the Institute of Industrial Applications Engineers*, 13, 11–20.
