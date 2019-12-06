package paristech

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{CountVectorizer, IDF, OneHotEncoderEstimator, RegexTokenizer, StopWordsRemover, StringIndexer, VectorAssembler}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}

object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Trainer")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    /*******************************************************************************
      *
      *       TP 3
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    // lecture du fichier Parquet

    val df = spark.read
      .option("header", true)
      .option("inferSchema", "true")
      .parquet("src/main/resources/preprocessed_clean")

    // Creation du pipeline avec la fonction createPipeline(DataFrame)

    val (pipeline, cv, lr) = createPipeline(df)

    // Separation Test / Train

    val Array(training, test) = df.randomSplit(Array(0.9, 0.1))


    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")

    // La prédiction affichée peut être la prédiction simple avec la fonction simplePrediction
    // Ou celle obtenue avec le réglage des hyperparamètres

    //simplePredictions(training, test, pipeline, evaluator)

    hyperparameterPrediction(cv, lr, pipeline, evaluator, training, test)

  }

  def createPipeline(df : DataFrame): (Pipeline , CountVectorizer, LogisticRegression) ={
    // Etape 1 : tokenizer

    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    // Etape 2 : remover

    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("tokens_without_sw")

    // Etape 3 : CountVectorizer

    val cv = new CountVectorizer()
      .setInputCol("tokens_without_sw")
      .setOutputCol("tf")
      .setMinDF(3)

    // Etape 4 : IDF

    val idf = new IDF().setInputCol("tf").setOutputCol("tfidf")

    // Etape 5 : country_indexer

    val country_indexer = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")
      .setHandleInvalid("keep")

    // Etape 6 : currency_indexer

    val currency_indexer = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")
      .setHandleInvalid("keep")

    // Etapes 7 et 8 : encoder

    val encoder = new OneHotEncoderEstimator()
      .setInputCols(Array("country_indexed", "currency_indexed"))
      .setOutputCols(Array("country_onehot", "currency_onehot"))

    // Etape 9 : Assembler

    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_onehot", "currency_onehot"))
      .setOutputCol("features")

    // Etape 10 : Model

    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(20)

    // Creation du Pipeline

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, cv, idf, country_indexer, currency_indexer, encoder, assembler, lr))

    return (pipeline, cv, lr)

  }

  def simplePredictions(training: DataFrame, test: DataFrame, pipeline: Pipeline, evaluator: MulticlassClassificationEvaluator): Unit ={

    val model = pipeline.fit(training)

    // Sauvegarde du modèle

    model.write.overwrite().save("src/main/models/logistic-regression-model")

    // Résultat du modèle sur l'échantillon de test

    val dfWithSimplePredictions = model.transform(test)
    dfWithSimplePredictions.groupBy("final_status", "predictions").count.show()

    // Calcul de f1

    val f1 = evaluator.evaluate(dfWithSimplePredictions)
    println("Test set f1 = " + f1)
  }

  def hyperparameterPrediction(cv: CountVectorizer, lr: LogisticRegression, pipeline: Pipeline,
                 evaluator: MulticlassClassificationEvaluator, training : DataFrame, test: DataFrame): Unit ={

    // Creation de la grille de recherche pour ajuster les hyperparamètres

    val paramGrid = new ParamGridBuilder()
      .addGrid(cv.minDF, Array(55.0, 75.0, 95.0))
      .addGrid(lr.tol, Array(10e-8, 10e-6, 10e-4, 10e-2))
      .build()

    // Reglage des hyperparamètres

    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)

    // Selection du meilleur modèle

    val bestModel = trainValidationSplit.fit(training)

    // Sauvegarde du modèle

    bestModel.write.overwrite().save("src/main/models/logistic-regression-bestmodel")

    // Résultat du modèle sur l'échantillon de test

    val dfWithPredictions = bestModel.transform(test)
    dfWithPredictions.groupBy("final_status", "predictions").count.show()

    // Calcul de f1

    val f1Best = evaluator.evaluate(dfWithPredictions)
    println("Test set f1 = " + f1Best)
  }
}
