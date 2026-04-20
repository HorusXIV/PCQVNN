---
bibliography: ./report/Citations.bib
csl: apa.csl 
link-citations: true
---

# Quaternion-Valued Neural Networks and Predictive Coding

## Projektidee

In diesem Projekt möchte ich untersuchen, ob ein quaternionwertiges neuronales Netzwerk (QVNN) mit Predictive Coding (PC) vernünftig trainiert werden kann und wie sich ein solches Modell bei Standardaufgaben zur Approximation nichtlinearer Funktionen im Vergleich zu einem starken Quaternion-Backpropagation-Basismodell schlägt.  

Konkret besteht das Ziel darin, die Benchmark-Einstellung von Bill, Cox und Champagne [@BILL2023120448] zu erweitern, indem die Quaternion-Backpropagation durch ein auf Predictive Coding basierendes Trainingsverfahren ersetzt und beide Ansätze unter derselben Architektur und denselben Benchmark-Funktionen verglichen werden. 

## Quaternion-Valued Neural Networks (QVNN)

Ein quaternionwertiges neuronales Netzwerk (QVNN) ist ein neuronales Netzwerk, in dem Eingaben, Gewichte und Aktivierungen nicht als reelle Zahlen, sondern als **Quaternionen** dargestellt werden. Ein Quaternion hat vier Komponenten und kann als
$$
q = a + b\mathbf{i} + c\mathbf{j} + d\mathbf{k}
$$
geschrieben werden. Im Gegensatz zu reellen Zahlen ist die Quaternionenmultiplikation über das Hamilton-Produkt nicht kommutativ, also gilt im Allgemeinen $q_1 \otimes q_2 \neq q_2 \otimes q_1$. Deswegen sind QVNNs schwieriger zu trainieren als gewöhnliche reellwertige Netze, bieten allerdings auch einige Vorteile. 

Dadurch dass, das Neuron durch vier miteinander Verknüpften Werte besteht, bietet ein QVNN besonders dann Vorteile, wenn Daten aus vier zusammenhängenden Komponenten bestehen (z.B. RGBA, 3D Rotationen, Signalverarbeitung). Die Literatur zeigt zudem, dass Quaternion-Netzwerke in verschiedenen Anwendungen kompaktere Modelle und teils bessere Generalisierung als vergleichbare reellwertige Netze liefern können [@article]. Für Quaternion-Netze existiert jedoch keine einheitliche Standard-Backpropagation; stattdessen gibt es mehrere Varianten wie Arena, Nitta, CMLP und GHR [@math9090938].  

## Predictive Coding (PC)

Predictive Coding ist ein alternatives Lernprinzip zu klassischer Backpropagation. Die Grundidee ist, dass ein Modell nicht  nur vorwärts rechnet, sondern auf jeder Ebene versucht, die darunterliegende Aktivität vorherzusagen. Gelernt wird dann über die Minimierung von Vorhersagefehlern. Dabei werden zunächst latente Zustände iterativ angepasst (Inferenz) und anschliessend die Gewichte aktualisiert (lernen). Das führt zu einer charakteristischen Dynamik aus schneller Inferenz und langsamerem Gewichtslernen [@stenlund2025introductionpredictivecodingnetworks]. 

Ein wesentlicher Vorteil von PC ist, dass die Gewichtsupdates lokal formuliert sind: Sie hängen nur von Aktivität und Fehlern benachbarter Schichten ab. Dadurch ist PC konzeptionell näher an biologisch inspirierten Lernmechanismen und zugleich interessant für dezentrale oder parallele Rechenstrukturen. Ausserdem erlaubt PC prinzipiell adaptive Inferenz, bei der einfache Beispiele mit wenigen Schritten auskommen, während schwierigere Beispiele mehr Iterationen benötigen [@stenlund2025introductionpredictivecodingnetworks].  

## Projekt

Ich möchte diese beiden Ansätze kombinieren und den Vergleich von @BILL2023120448 um ein Modell erweitern, das mit einem PC-Algorithmus optimiert wurde. Aufgrund der Eigenschaften der Quaternionenalgebra $R$⁴ sind viele Standardfunktionen wie ReLU usw. nicht differenzierbar, was alles viel schwieriger macht. Ich erhoffe mir deshalb, mit diesem alternativen Ansatz die langen Produktregel-Verkettungen bei der Ableitung zu vermeiden. Eventuell kann ich durch entsprechende Parallelisierung auch Geschwindigkeitsvorteile erzielen.

Die Architektur, die ich verwenden möchte, ist die gleiche wie im Paper: 1-5-5-5-1. PC hat bekanntermassen Probleme bei tiefen neuronalen Netzwerken, aber das sollte flach genug sein, sodass das keine weitere Schwierigkeit darstellt.

Es soll untersucht werden, ob ein mit PC trainiertes QVNN bei der Approximation nichtlinearer Funktionen sinnvoll funktioniert und wie es sich gegenüber einem starken Quaternion-Backpropagation-Basismodell verhält.   
Als Vergleichs-Baseline bietet sich dabei insbesondere CMLP an, da diese Methode in der Vergleichsstudie die überzeugendsten Resultate unter den Quaternion-Backpropagation-Verfahren erzielt hat.  

# References