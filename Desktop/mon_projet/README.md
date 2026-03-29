# Projet Deep Learning : CNN & LSTM

## Objectif Général

Ce projet a pour objectif de passer de la théorie mathématique à une implémentation complète en Deep Learning.

Il couvre deux domaines :

*  Vision par Ordinateur (CNN)
*  Séries Temporelles (LSTM)

Le projet suit une architecture modulaire conforme aux standards de l’industrie.

---

#  Partie 1 : Classification d’Images (CNN)

##  Dataset

* CIFAR-10
* 60 000 images (32×32)
* 10 classes

---

##  Prétraitement

* Normalisation des pixels (0 → 1)
* Data Augmentation :

  * RandomFlip
  * RandomRotation

---

##  Architecture

* 3 couches Conv2D (3×3)
* MaxPooling2D
* Flatten
* Dense
* Dropout

Optimisation :

* Optimizer : Adam
* Loss : SparseCategoricalCrossentropy

---

##  Résultats CNN

* Train Accuracy : ~82%
* Validation Accuracy : ~81%
* Test Accuracy : **81.39%**
* Test Loss : 0.5673
* Meilleure epoch : 33 / 50

Objectif (70%) dépassé (+11.39%)

---

#  Partie 2 : Prédiction Temporelle (LSTM)

##  Dataset

* Jena Climate Dataset

---

##  Prétraitement

* Normalisation (MinMaxScaler)
* Création de séquences temporelles (Sliding Window)

---

##  Architecture

* LSTM
* return_sequences = False
* Dense final

Optimisation :

* Loss : MSE

---

##  Résultats LSTM

* RMSE : 0.5456 °C
* MAE : 0.4355 °C
* Meilleure epoch : 23

---

#  Structure du projet

```
.
├── data/              
├── models/            
├── outputs/           
├── saved_model/       
├── utils/             

├── app.py             
├── train.py           
├── train_lstm.py      
├── evaluate.py        
├── evaluate_lstm.py   

├── README.md          
└── requirements.txt   
```

---

#  Organisation

* **data/** : datasets (CIFAR-10, Jena)
* **models/** : architectures CNN et LSTM
* **utils/** : preprocessing, transformation, visualisation
* **outputs/** : résultats (graphes, métriques)
* **saved_model/** : modèles entraînés sauvegardés

---

#  Installation

```
pip install -r requirements.txt
```

---

#  Utilisation

### 🔹 Entraînement

```
python train.py
python train_lstm.py
```

### 🔹 Évaluation

```
python evaluate.py
python evaluate_lstm.py
```

### 🔹 Application 

```
python app.py
```

---

#  Résultats Globaux

| Modèle | Performance     |
| ------ | --------------- |
| CNN    | 81.39% accuracy |
| LSTM   | RMSE = 0.5456   |

---

# Points Forts

* Architecture modulaire (niveau professionnel)
* Implémentation CNN et LSTM complète
* Bonne généralisation des modèles
* Objectifs atteints et dépassés

---

#  Livrables

* Code structuré et documenté
* Modèles entraînés
* Graphiques (loss, prédictions)

---

#  Auteur

* TCHATCHOUANG DJUICHI AELLE
* NOUMEDEM-MEGNIKENG -NERGELO
* KUEKAM GOULAH KIRIANE LA FORTUNE
* TCHAPGA TOUMI NADEGE SANDRA
* ONDOBO ENAMA PATRICIA LEANDRA

---
