package paristech

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window

object Preprocessor {

  def main(args: Array[String]): Unit = {

    // Des réglages optionnels du job spark. Les réglages par défaut fonctionnent très bien pour ce TP.
    // On vous donne un exemple de setting quand même
    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
    ))

    // Initialisation du SparkSession qui est le point d'entrée vers Spark SQL (donne accès aux dataframes, aux RDD,
    // création de tables temporaires, etc., et donc aux mécanismes de distribution des calculs)
    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Preprocessor")
      .getOrCreate()


    spark.sparkContext.setLogLevel("WARN")

    import spark.implicits._

    /*******************************************************************************
      *
      *       TP 2
      *
      *       - Charger un fichier csv dans un dataFrame
      *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
      *       - Sauver le dataframe au format parquet
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    // Chargement fichier .csv
    val df = spark
      .read
      .option("header", true) // utilise la première ligne du (des) fichier(s) comme header
      .option("inferSchema", "true") // pour inférer le type de chaque colonne (Int, String, etc.)
      .csv("src/main/resources/train/train_clean.csv")

    println(s"Nombre de lignes : ${df.count}")
    println(s"Nombre de colonnes : ${df.columns.length}")

    df.printSchema()

    // Cast des colonnes

    val dfCasted = df
      .withColumn("goal", $"goal".cast("Int"))
      .withColumn("deadline" , $"deadline".cast("Int"))
      .withColumn("state_changed_at", $"state_changed_at".cast("Int"))
      .withColumn("created_at", $"created_at".cast("Int"))
      .withColumn("launched_at", $"launched_at".cast("Int"))
      .withColumn("backers_count", $"backers_count".cast("Int"))
      .withColumn("final_status", $"final_status".cast("Int"))

    val dfCasted1 = dfCasted.groupBy("disable_communication").count.orderBy($"count".desc)
    val dfCasted2 = dfCasted.groupBy("country").count.orderBy($"count".desc)
    val dfCasted3 = dfCasted.groupBy("currency").count.orderBy($"count".desc)
    val dfCasted4 = dfCasted.select("deadline").dropDuplicates
    val dfCasted5 = dfCasted.groupBy("state_changed_at").count.orderBy($"count".desc)
    val dfCasted6 = dfCasted.groupBy("backers_count").count.orderBy($"count".desc)
    val dfCasted7 = dfCasted.select("goal", "final_status")
    val dfCasted8 = dfCasted.groupBy("country", "currency").count.orderBy($"count".desc)

    // Somme cummulée des counts de chaques colonnes

    val nRows = df.count

    dfCasted1.withColumn("cumulativeSum", sum(dfCasted1("count")/nRows*100).over( Window.orderBy($"count".desc))).show(10)
    dfCasted2.withColumn("cumulativeSum", sum(dfCasted2("count")/nRows*100).over( Window.orderBy($"count".desc))).show(10)
    dfCasted3.withColumn("cumulativeSum", sum(dfCasted3("count")/nRows*100).over( Window.orderBy($"count".desc))).show(10)
    dfCasted4.show()
    dfCasted5.withColumn("cumulativeSum", sum(dfCasted5("count")/nRows*100).over( Window.orderBy($"count".desc))).show(10)
    dfCasted6.withColumn("cumulativeSum", sum(dfCasted6("count")/nRows*100).over( Window.orderBy($"count".desc))).show(10)
    dfCasted7.show(30)
    dfCasted8.withColumn("cumulativeSum", sum(dfCasted8("count")/nRows*100).over( Window.orderBy($"count".desc))).show(10)

    //retrait des données du futur

    val df2 = dfCasted.drop("disable_communication")
    val dfNoFutur = df2.drop("backers_count", "state_changed_at")

    // UDF pour nettoyer les colonnes country et currency et lever les incohérences

    def cleanCountry(country: String, currency: String): String = {
      if (country == "False")
        currency
      else
        country
    }

    def cleanCurrency(currency: String): String = {
      if (currency != null && currency.length != 3)
        null
      else
        currency
    }

    val cleanCountryUdf = udf(cleanCountry _)
    val cleanCurrencyUdf = udf(cleanCurrency _)

    val dfCountry = dfNoFutur
      .withColumn("country2", cleanCountryUdf($"country", $"currency"))
      .withColumn("currency2", cleanCurrencyUdf($"currency"))
      .drop("country", "currency")


    var dfFinalStatus = dfCountry.filter($"final_status" === 1 || $"final_status" === 0).groupBy("final_status").count

    dfFinalStatus.withColumn("cumulativeSum", sum(dfFinalStatus("count")/nRows*100).over( Window.orderBy($"count".desc))).show()

    // Il y a 0.41% des final_status qui n'ont ni le label 0 ou 1. Ces données sont supprimées
    // afin d'équilibrer les valeurs de notre classe cible

    dfFinalStatus = dfCountry.filter($"final_status" === 1 || $"final_status" === 0)

    // Ajout d'une colonne days_campaign qui représente la durée de la campagne en jours
    // (le nombre de jours entre launched_at et deadline)

    val dfDaysCampaign = dfFinalStatus.withColumn("days_campaign", ($"deadline" - $"launched_at")/(3600*24))
    dfDaysCampaign.select("days_campaign", "deadline" , "launched_at").orderBy($"days_campaign".desc).show(3)

    // Ajout d'une colonne hours_prepa qui représente
    // le nombre d’heures de préparation de la campagne entre created_at et launched_at

    val dfDaysPrep = dfDaysCampaign.withColumn("hours_prepa", ($"launched_at" - $"created_at")/3600)
    dfDaysPrep.select("hours_prepa", "launched_at" , "created_at").orderBy($"hours_prepa".desc).show(3)

    // Suppression des colonnes inutiles au training
    dfDaysPrep.drop("launched_at", "created_at", "deadline")

    // Concatenation des colonnes contenant du texte

    val dfClean = dfDaysPrep.withColumn("text", concat($"name", lit(" "), $"desc", lit(" "), $"keywords"))

    dfClean.na.fill(-1, Seq("days_campaign"))
            .na.fill(-1, Seq("hours_prepa"))
              .na.fill(-1, Seq("goal"))
                .na.fill("unknown", Seq("country2"))
                    .na.fill("unknown", Seq("currency2"))

    // Ecriture du dataframe en Parquet

    dfClean.show(2)

    dfClean.write.mode("overwrite").parquet("src/main/resources/preprocessed")

  }
}
