# Quaternion-Valued Neural Networks and Predictive Coding

## Projektidee

In diesem Projekt möchte ich untersuchen, ob ein quaternionwertiges neuronales Netzwerk (QVNN) mit Predictive Coding (PC) vernünftig trainiert werden kann und wie sich ein solches Modell bei Standardaufgaben zur Approximation nichtlinearer Funktionen im Vergleich zu einem starken Quaternion-Backpropagation-Basismodell schlägt.  

Konkret besteht das Ziel darin, die Benchmark-Einstellung von Bill, Cox und Champagne zu erweitern, indem die Quaternion-Backpropagation durch ein auf Predictive Coding basierendes Trainingsverfahren ersetzt und beide Ansätze unter derselben Architektur und denselben Benchmark-Funktionen verglichen werden. 

## Quaternion-Valued Neural Networks (QVNN)

Ein quaternionwertiges neuronales Netzwerk (QVNN) ist ein neuronales Netzwerk, in dem Eingaben, Gewichte und Aktivierungen nicht als reelle Zahlen, sondern als **Quaternionen** dargestellt werden. Ein Quaternion hat vier Komponenten und kann als
$$
q = a + b\mathbf{i} + c\mathbf{j} + d\mathbf{k}
$$
geschrieben werden. Im Gegensatz zu reellen Zahlen ist die Quaternionenmultiplikation über das Hamilton-Produkt nicht kommutativ, also gilt im Allgemeinen $q_1 \otimes q_2 \neq q_2 \otimes q_1$. Genau diese algebraische Struktur macht QVNNs mathematisch interessant, aber auch schwieriger zu trainieren als gewöhnliche reellwertige Netze. 

QVNNs sind besonders dann attraktiv, wenn Daten aus vier zusammenhängenden Komponenten bestehen, da diese direkt in einem Quaternion gebündelt werden können. Die Literatur zeigt zudem, dass Quaternion-Netzwerke in verschiedenen Anwendungen kompaktere Modelle und teils bessere Generalisierung als vergleichbare reellwertige Netze liefern können. Für Quaternion-Netze existiert jedoch keine einheitliche Standard-Backpropagation; stattdessen gibt es mehrere Varianten wie Arena, Nitta, CMLP und GHR.  

## Predictive Coding (PC)

Predictive Coding ist ein alternatives Lernprinzip zu klassischer Backpropagation. Die Grundidee ist, dass ein Modell nicht einfach nur vorwärts rechnet, sondern auf jeder Ebene versucht, die darunterliegende Aktivität vorherzusagen. Gelernt wird dann über die Minimierung von Vorhersagefehlern. Dabei werden zunächst latente Zustände iterativ angepasst (Inferenz) und anschließend die Gewichte aktualisiert (lernen). Das führt zu einer charakteristischen Dynamik aus schneller Inferenz und langsamerem Gewichtslernen. 

Ein wesentlicher Vorteil von PC ist, dass die Gewichtsupdates lokal formuliert sind: Sie hängen nur von Aktivität und Fehlern benachbarter Schichten ab. Dadurch ist PC konzeptionell näher an biologisch inspirierten Lernmechanismen und zugleich interessant für dezentrale oder parallele Rechenstrukturen. Außerdem erlaubt PC prinzipiell adaptive Inferenz, bei der einfache Beispiele mit wenigen Schritten auskommen, während schwierigere Beispiele mehr Iterationen benötigen.  

## PCQVNN

Die Kombination aus QVNN und Predictive Coding ist für mich interessant: eine strukturreiche hyperkomplexe Darstellung auf der einen Seite und ein nicht-standardmäßiges Lernverfahren auf der anderen. QVNNs sind vielversprechend, aber ihr Training ist wegen der Nichtkommutativität der Quaternionenalgebra und der verschiedenen konkurrierenden Backpropagation-Regeln oft recht instabil. Besonders wiel beide , ob Predictive Coding in diesem Kontext eine praktikable Alternative zu Quaternion-Backpropagation sein kann. 

Das Ziel des Projekts ist daher nicht zu zeigen, dass Predictive Coding besser ist. Stattdessen soll untersucht werden, ob ein mit PC trainiertes QVNN bei der Approximation nichtlinearer Funktionen sinnvoll funktioniert und wie es sich gegenüber einem starken Quaternion-Backpropagation-Basismodell verhält.   
Als Baseline bietet sich dabei insbesondere CMLP an, da diese Methode in der Vergleichsstudie die überzeugendsten Resultate unter den Quaternion-Backpropagation-Verfahren erzielt hat.  
