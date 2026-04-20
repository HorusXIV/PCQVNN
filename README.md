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

### Baseline & Evaluation
Als primäre Baseline dient das CMLP (Complex Multi-Layer Perceptron)-Verfahren, das in der Vergleichsstudie von Bill et al. die besten Resultate lieferte. Evaluiert wird die Performance bei der Approximation nichtlinearer Funktionen hinsichtlich:

* Approximationsgenauigkeit (MSE)
* Konvergenzgeschwindigkeit
* Recheneffizienz

## References

*   Bill, J., Champagne, L., Cox, B., & Bihl, T. (2021). [Meta-Heuristic Optimization Methods for Quaternion-Valued Neural Networks](https://doi.org). *Mathematics*, 9(938).
*   Bill, J., Cox, B. A., & Champagne, L. (2023). [A Comparison of Quaternion Neural Network Backpropagation Algorithms](https://doi.org). *Expert Systems with Applications*, 232, 120448.
*   Stenlund, M. (2025). [Introduction to Predictive Coding Networks for Machine Learning](https://arxiv.org/abs/2506.06332). *arXiv preprint arXiv:2506.06332*.
*   Takahashi, K., Fujimoto, A., Kawakami, S., Mitera, H., & Hashimoto, M. (2025). [Optimal Predictive Control Using Quaternion-Valued Neural Network-Based System Identification](https://doi.org). *Journal of the Institute of Industrial Applications Engineers*, 13, 11–20.
