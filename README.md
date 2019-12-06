# Spark project MS Big Data Télécom : Kickstarter campaigns


* Etapes pour cloner le repo GitHub et aller dans le bon dossier:

```bash
git clone git@github.com:francouee/spark_project_kickstarter_2019_2020.git
cd spark_project_kickstarter_2019_2020 
```

* Il faut avant de lancer le ```build_and_submit.sh``` créer une variable d'environnement ```$SPARK_HOME``` qui donne l'emplacement de SPARK sur votre ordinateur.

* Pour le preprocessing du jeu de données:

```bash
./build_and_submit.sh Preprocessor $SPARK_HOME 
```

* Pour l'entrainement du modèle:

```bash
./build_and_submit.sh Trainer $SPARK_HOME
```
