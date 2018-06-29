package com.pavanpkulkarni.model

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.{Encoders, SparkSession}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}


object CreateAndSaveModel_Pipeline {
	
	case class Credit(creditability: Double, balance: Double, duration: Double, history: Double, purpose: Double, amount: Double, savings: Double, employment: Double, instPercent: Double, sexMarried: Double, guarantors: Double, residenceDuration: Double, assets: Double, age: Double, concCredit: Double, apartment: Double, credits: Double, occupation: Double, dependents: Double, hasPhone: Double, foreign: Double)
	
	def main(args: Array[String]): Unit = {
		
		val spark = SparkSession
			.builder()
			.master("local") //uncomment this line when running on local
			.getOrCreate()
		
		import spark.implicits._
		
		val data = spark
			.read
			.schema(Encoders.product[Credit].schema)
			.csv("DS_CreateAndSaveModel/src/main/resources/index.csv")
			.as[Credit]
  		//.select("balance", "duration", "history", "purpose","creditability")
		
		val labelIndexer = new StringIndexer()
			.setInputCol("creditability")
			.setOutputCol("iLabel")
			.fit(data)
		
		val featureCols = Array("balance", "duration", "history", "purpose", "amount","savings", "employment", "instPercent" ,"sexMarried",  "guarantors",	"residenceDuration", "assets",  "age", "concCredit", "apartment",	"credits",  "occupation", "dependents",  "hasPhone", "foreign" )
		
		//val featureCols = Array("balance", "duration", "history", "purpose" )
		
		val featureAssembler = new VectorAssembler()
			.setInputCols(featureCols)
			.setOutputCol("assembledFeatures")
		
		val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
		
		val rf = new RandomForestClassifier()
			.setLabelCol("iLabel")
			.setFeaturesCol("assembledFeatures")
			.setNumTrees(10)
		
		val pipeline = new Pipeline()
			.setStages(Array(labelIndexer, featureAssembler, rf))
		
		val model = pipeline.fit(trainingData)
		
		val predictions = model.transform(testData)

		predictions.show(false)
		
		predictions.select("creditability", "probability","prediction").show(false)
		
		val evaluator = new MulticlassClassificationEvaluator()
			.setLabelCol("iLabel")
			.setPredictionCol("prediction")
			.setMetricName("accuracy")
		
		val accuracy = evaluator.evaluate(predictions)
		println(s"Test Error = ${(1.0 - accuracy)}")
		
		// Model Tuning
		
		val paramGrid = new ParamGridBuilder()
			.addGrid(rf.maxBins, Array(25, 28, 31))
			.addGrid(rf.maxDepth, Array(4, 6, 8))
			.addGrid(rf.impurity, Array("entropy", "gini"))
			.build()
		
		val cv = new CrossValidator()
			.setEstimator(pipeline)
			.setEvaluator(evaluator)
			.setEstimatorParamMaps(paramGrid)
			.setParallelism(9)
		
		val tunedModel = cv.fit(trainingData)
		val tunedPredictions = tunedModel.transform(testData)
		tunedPredictions.select("creditability", "probability","prediction").show(false)
		val tunedAccuracy = evaluator.evaluate(tunedPredictions)
		
		println(s"Test Error = ${(1.0 - tunedAccuracy)}")
		
		//Save the tuned model
		tunedModel.write.overwrite().save("DS_CreateAndSaveModel/src/main/resources/RandomForestModel")

		
	}
}
