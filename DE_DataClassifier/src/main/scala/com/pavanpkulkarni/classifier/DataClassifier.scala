package com.pavanpkulkarni.classifier


import org.apache.spark.sql.{Encoders, SparkSession}
import org.apache.spark.ml.PipelineModel


object DataClassifier {
	
	case class Credit(creditability: Double, balance: Double, duration: Double, history: Double, purpose: Double, amount: Double, savings: Double, employment: Double, instPercent: Double, sexMarried: Double, guarantors: Double, residenceDuration: Double, assets: Double, age: Double, concCredit: Double, apartment: Double, credits: Double, occupation: Double, dependents: Double, hasPhone: Double, foreign: Double )
	
	def main(args: Array[String]): Unit = {
		
		
		val spark = SparkSession
			.builder()
			.master("local")
			.appName("Data_Classifier_Streaming")
			.getOrCreate()
		
		val model = PipelineModel.load("DS_CreateAndSaveModel/src/main/resources/RandomForestModel")
		
		import spark.implicits._
		
		val creditDS = spark
			.readStream
			.schema(Encoders.product[Credit].schema)
			.csv("DE_DataClassifier/src/main/resources/input")
			.as[Credit]
			.select("balance", "duration", "history", "purpose","creditability")
		
		val predictions = model.transform(creditDS)
		
		val finalPredictions = predictions.select("creditability","probability","prediction")
		
		val query = finalPredictions
			.writeStream
			.queryName("count_customer")
			//.format("console")
			.outputMode("append")
			.format("json")
			//.partitionBy("date")
			.option("path", "DE_DataClassifier/src/main/resources/output/")
			.option("checkpointLocation", "DE_DataClassifier/src/main/resources/chkpoint_dir")
			.start()
		
		query.awaitTermination()
		
	}
}
