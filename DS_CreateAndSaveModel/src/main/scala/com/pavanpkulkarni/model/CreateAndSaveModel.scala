package com.pavanpkulkarni.model

import org.apache.spark.sql.{Dataset, Encoders, SparkSession}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.rdd.RDD

object CreateAndSaveModel {
	
	case class Credit(creditability: Double, balance: Double, duration: Double, history: Double, purpose: Double, amount: Double, savings: Double, employment: Double, instPercent: Double, sexMarried: Double, guarantors: Double, residenceDuration: Double, assets: Double, age: Double, concCredit: Double, apartment: Double, credits: Double, occupation: Double, dependents: Double, hasPhone: Double, foreign: Double)
	
	
	def main(args: Array[String]): Unit = {
		
		val spark = SparkSession
			.builder()
			.master("local") //uncomment this line when running on local
			.getOrCreate()
		
		import spark.implicits._
		
		val creditDS = spark
			.read
			.schema(Encoders.product[Credit].schema)
			.csv("DS_CreateAndSaveModel/src/main/resources/index.csv")
			.as[Credit]
		
		//creditDS.printSchema()
		
		//creditDS.describe("balance").show(false)
		
		val featureCols = Array("balance", "duration", "history", "purpose", "amount","savings", "employment", "instPercent" ,"sexMarried",  "guarantors",	"residenceDuration", "assets",  "age", "concCredit", "apartment",	"credits",  "occupation", "dependents",  "hasPhone", "foreign" )
		
		val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
		val df2 = assembler.transform(creditDS)
		//df2.show(false)
		
		val labelIndexer = new StringIndexer().setInputCol("creditability").setOutputCol("label")
		val df3 = labelIndexer.fit(df2).transform(df2)
		//df3.show(false)
		
		val splitSeed = 5043
		val Array(trainingData, testData) = df3.randomSplit(Array(0.7, 0.3), splitSeed)
		
		//testData.write.json("DS_CreateAndSaveModel/src/main/resources/testData.csv")
		
		println("count of Training Set : ", trainingData.count())
		
		println("count of Testing Set : ", testData.count())
		
		val classifier = new RandomForestClassifier()
			.setImpurity("gini")
			.setMaxDepth(3)
			.setNumTrees(20)
			.setFeatureSubsetStrategy("auto")
			.setSeed(5043)
		
		val RandomForestModel = classifier.fit(trainingData)
		
		//println("Debugging the model : ", RandomForestModel.toDebugString)
		
		val predictions = RandomForestModel.transform(testData)
		
		predictions.select("creditability","label","rawPrediction","probability","prediction")
  		.where("label = 0.0")
			.show(false)
		
		predictions.select("creditability","label","rawPrediction","probability","prediction")
			.where("label = 1.0")
			.show(false)
		
		val evaluator = new BinaryClassificationEvaluator().setLabelCol("label")
		val accuracy = evaluator.evaluate(predictions)
		
		println("Accuracy : ", accuracy)
		
		val paramGrid = new ParamGridBuilder()
			.addGrid(classifier.maxBins, Array(25, 28, 31))
			.addGrid(classifier.maxDepth, Array(4, 6, 8))
			.addGrid(classifier.impurity, Array("entropy", "gini"))
			.build()
		
		val steps: Array[PipelineStage] = Array(classifier)
		val pipeline = new Pipeline().setStages(steps)
		
		RandomForestModel.save("DS_CreateAndSaveModel/src/main/resources/RandomForestModel")
		
		spark.stop()
		
	}
}
