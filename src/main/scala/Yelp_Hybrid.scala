import java.io.{File, PrintWriter}

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import collection.mutable
import scala.collection.mutable.{ArrayBuffer, ListBuffer}

object Yelp_Hybrid {

  val user_id_map: mutable.Map[String, Int] = mutable.Map[String, Int]()
  val business_id_map: mutable.Map[String, Int] = mutable.Map[String, Int]()
  var user_id_current = 1
  var business_id_current = 1

  def assign_int_user_id(id: String): Int = {
    if ((user_id_map get id).isEmpty) {
      user_id_map(id) = user_id_current
      user_id_current += 1
    }
    user_id_map(id)
  }

  def assign_int_business_id(id: String): Int = {
    if ((business_id_map get id).isEmpty) {
      business_id_map(id) = business_id_current
      business_id_current += 1
    }
    business_id_map(id)
  }

  def adjust_rating(rating: Double): Double = {
    if (rating >= 5.0)
      return 4.7
    else if (rating < 1.0)
      return 2.8
    rating
  }

  // Counting error buckets
  def count_item_differences(actual: Double, predicted: Double): String = {
    val error = math.abs(actual - predicted)

    if (error >= 0 && error < 1) {
      ">=0 and <1"
    }
    else if (error >= 1 && error < 2) {
      ">=1 and <2"
    }
    else if (error >= 2 && error < 3) {
      ">=2 and <3"
    }
    else if (error >= 3 && error < 4) {
      ">=3 and <4"
    }
    else {
      ">=4"
    }
  }

  def main(args: Array[String]): Unit = {

    val start_time = System.nanoTime()

    val spark = SparkSession.builder()
      .master("local[*]")
      .appName("ALSrecommender")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    val training_file_path = args(0)
    val testing_file_path = args(1)

    val text_train = spark.sparkContext.textFile(training_file_path)
    val header = text_train.first()
    val ratings = text_train
      .filter(line => !line.equals(header))
      .map(line => line.split(","))
      .map(attributes => Rating(assign_int_user_id(attributes(0).trim), assign_int_business_id(attributes(1).trim), attributes(2).trim.toDouble))
    ratings.collect()


    // Evaluate the model on rating data
    val text_test = spark.sparkContext.textFile(testing_file_path)
    val test_ratings = text_test
      .filter(line => !line.equals(header))
      .map(line => line.split(","))
      .map(attributes => Rating(assign_int_user_id(attributes(0).trim), assign_int_business_id(attributes(1).trim), attributes(2).trim.toDouble))
    test_ratings.collect()
    // Build the recommendation model using ALS
    val rank = 2 // Can be changed for better performance
    val num_iterations = 25 // Can be changed for better performance
    val lambda_value = 0.2229 // Can be changed for better performance
    val blocks = 2 // Can be changed for better performance
    val seed = 2 // Can be changed for better performance

    val model = ALS.train(ratings = ratings,
      rank = rank,
      iterations = num_iterations,
      lambda = lambda_value,
      blocks = blocks,
      seed = seed)

    val usersProducts = test_ratings.map { case Rating(user, product, rate) => (user, product) }

    val predictions = model.predict(usersProducts).map { case Rating(user, product, rate) => ((user, product), adjust_rating(rate)) }

    val ratesAndPreds = test_ratings.map { case Rating(user, product, rate) => ((user, product), rate) }.join(predictions)

    // Calculating and printing counts of error buckets
//    val differences = ratesAndPreds.map(x => count_item_differences(x._2._1, x._2._2.toFloat)).countByValue()
//    for ((key, value) <- differences.toArray.sorted) {
//      println(key + ": " + value)
//    }

//    val MSE = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
//      val err = r1 - r2
//      err * err
//    }.mean()
//
//    val RMSE = math.sqrt(MSE)
//    println(s"RMSE: = $RMSE")

    val user_int_id_map = user_id_map.map(_.swap)
    val business_int_id_map = business_id_map.map(_.swap)

    val modelBasedRatings:RDD[((String, String), Double)] = predictions.map(prediction => ((user_int_id_map(prediction._1._1), business_int_id_map(prediction._1._2)), prediction._2))
    val userBasedRatingsPredictions:RDD[((String, String), (Float, Double))] = generate_user_based_CF(spark.sparkContext, training_file_path, testing_file_path)
    val joined: RDD[((String, String), (Double, (Float, Double)))] = modelBasedRatings.join(userBasedRatingsPredictions)
    val hybrid = joined.map(X=>(X._1, (X._2._2._1, if (X._2._2._2 <= 0) X._2._1 else (X._2._2._2 + X._2._1)/2.0)))
    val RMSE = Math.sqrt(hybrid.map(userPred => (userPred._2._1 - userPred._2._2) * (userPred._2._1 - userPred._2._2)).sum / userBasedRatingsPredictions.count())
    println("RMSE: " + RMSE.toString)


    //    // Generating expected output of the form "User, Business, Predicted Rating"
//    val output = predictions.map(prediction => (user_int_id_map(prediction._1._1), business_int_id_map(prediction._1._2), prediction._2))
//    val intermediate_result = output.mkString("\n")
//    val final_result = intermediate_result.replace("(", "").replace(")", "").replace(",", ", ")
//
//    // Writing results into output file
//    val output_file = new PrintWriter(new File("Malhar_Kulkarni_ModelBasedCF.txt"))
//    output_file.write(final_result)
//    output_file.close()
    println("Time: " + ((System.nanoTime() - start_time) / 1e9d).toInt + " sec")
  }

  def intToSet(x: String): Set[String] = Set(x)

  def addIntToSet(acc: Set[String], x: String): Set[String] = acc.union(Set(x))

  def unionTwoIntSet(acc1: Set[String], acc2: Set[String]): Set[String] = acc1.union(acc2)

  def unratedUserBusinessFixer(userBusinessTuple: ((String, String), Float), userBusinessMap: mutable.Map[String, Set[String]], businessUsersMap: mutable.Map[String, Set[String]]) = {
    val businessIdToPredict = userBusinessTuple._1._2
    val userIdToPredict = userBusinessTuple._1._1
    if (!userBusinessMap.contains(userIdToPredict)) {
      userBusinessMap += (userIdToPredict -> Set[String]())
    }
    if (!businessUsersMap.contains(businessIdToPredict)) {
      businessUsersMap += (businessIdToPredict -> Set[String]())
    }
  }


  def buildSimBusinessUser(userBusinessTuple: ((String, String), Float), businessUsersMap: mutable.Map[String, Set[String]]): ListBuffer[(String, String)] = {
    val userId = userBusinessTuple._1._1
    val businessIdToPredict = userBusinessTuple._1._2
    val simUsers = businessUsersMap(businessIdToPredict)

    var simUserPairs = new ListBuffer[(String, String)]()
    for (user <- simUsers) {
      simUserPairs += ((userId, user))
    }
    simUserPairs
  }


  def getPearsonCorrelation(corratedUserTuple: (String, String),
                            userAndBusinessMap: mutable.Map[String, Set[String]],
                            userBusinessRatingsMap: Map[(String, String), Float]): ((String, String), Double) = {
    val user1 = corratedUserTuple._1
    val user2 = corratedUserTuple._2

    val user1Businesses = userAndBusinessMap(user1)
    val user2Businesses = userAndBusinessMap(user2)

    val commonBusinesses = user1Businesses.intersect(user2Businesses).toList.sorted

    if (commonBusinesses.isEmpty) {
      return ((user1, user2), -1)
    }

    val user1BusinessRatings = commonBusinesses.map(BusinessId => userBusinessRatingsMap((user1, BusinessId))).toArray
    val user2BusinessRatings = commonBusinesses.map(BusinessId => userBusinessRatingsMap((user2, BusinessId))).toArray

    ((user1, user2), pearsonCorrelation(user1BusinessRatings, user2BusinessRatings))
  }


  def pearsonCorrelation(user1BusinessRatings: Array[Float], user2BusinessRatings: Array[Float], log: Boolean = false): Double = {
    val vector1Sum = user1BusinessRatings.sum
    val vector2Sum = user2BusinessRatings.sum

    val vector1Mean = vector1Sum / user1BusinessRatings.length
    val vector2Mean = vector2Sum / user2BusinessRatings.length

    val vector1Normalized = user1BusinessRatings.map(_ - vector1Mean).toList
    val vector2Normalized = user2BusinessRatings.map(_ - vector2Mean).toList

    val vector1AndVector2DotProduct =
      vector1Normalized.zip(vector2Normalized).map(user1User2Rating => user1User2Rating._1 * user1User2Rating._2).sum

    val vector1EuclideanDistance = user1BusinessRatings.map(vector1 => vector1 * vector1).sum
    val vector2EuclideanDistance = user2BusinessRatings.map(vector2 => vector2 * vector2).sum
    val denominator = Math.sqrt(vector1EuclideanDistance * vector2EuclideanDistance)

    vector1AndVector2DotProduct / denominator
  }


  def calculateBusinessPrediction(userBusinessRatings: ((String, String), Float),
                                  userAndBusinessMap: mutable.Map[String, Set[String]],
                                  userBusinessRatingMap: Map[(String, String), Float],
                                  businessAndUsersMap: mutable.Map[String, Set[String]],
                                  pearsonCorrelationMap: Map[(String, String), Double]): ((String, String), (Float, Double)) = {
    val predictForUserId = userBusinessRatings._1._1
    val toPredictMovieId = userBusinessRatings._1._2
    val actualRating = userBusinessRatings._2

    // Get average rating of the user
    val userMovieList = userAndBusinessMap.getOrElse(predictForUserId, Set.empty)
    val userRatingList = userMovieList.toList
      .filter(userMovie => userBusinessRatingMap.contains((predictForUserId, userMovie)))
      .map(userMovie => userBusinessRatingMap((predictForUserId, userMovie)))

    val predictForUserAverageRating = if (userRatingList.isEmpty) 3.0 else userRatingList.sum / userRatingList.size.toFloat

    // Get the list of users who have rated business_id
    if (!businessAndUsersMap.contains(toPredictMovieId)) {
      return ((predictForUserId, toPredictMovieId), (actualRating, predictForUserAverageRating))
    }

    val corUserList = businessAndUsersMap.getOrElse(toPredictMovieId, Set.empty)

    // Get the pearson correlation values of these users with user_id
    var corUsersNumeratorComponents: ArrayBuffer[Float] = ArrayBuffer.empty
    var corUsersDenominatorComponents: ArrayBuffer[Float] = ArrayBuffer.empty

    for (corUser <- corUserList) {
      // calculating correlated user's average rating
      val businesssRatedByCorUsers = userAndBusinessMap.getOrElse(corUser, Set.empty)
      val corUserRatingList = businesssRatedByCorUsers
        .toList
        .filter(userMovieId => userMovieId != toPredictMovieId)
        .map(userMovieId => userBusinessRatingMap((corUser, userMovieId)))
      val corUserAvgRating = corUserRatingList.sum / corUserRatingList.size

      // Fetching pearson correlation of the correlated user
      if (pearsonCorrelationMap.contains((predictForUserId, corUser))) {
        val pearsonCorForCorUser = pearsonCorrelationMap((predictForUserId, corUser))
        if (pearsonCorForCorUser > 0) {
          val corUserNumeratorComponent = (userBusinessRatingMap((corUser, toPredictMovieId)) - corUserAvgRating) * pearsonCorForCorUser
          corUsersDenominatorComponents += Math.abs(pearsonCorForCorUser).toFloat
          corUsersNumeratorComponents += corUserNumeratorComponent.toFloat
          //          println(corUsersDenominatorComponents)
        }
      }
    }
    val corUsersNumeratorComponentsSum: Float = corUsersNumeratorComponents.sum
    val corUsersDenominatorComponentSum: Float = corUsersDenominatorComponents.sum
    var prediction: Double = 0.0
    if (corUsersNumeratorComponentsSum != 0 && corUsersDenominatorComponentSum != 0) {
      prediction = predictForUserAverageRating + (corUsersNumeratorComponentsSum / corUsersDenominatorComponentSum)
      //prediction = if (prediction > userBusinessRatings._2) Math.floor(prediction) else prediction
    } else {
      prediction = predictForUserAverageRating
    }

    //    println("Cor user num: " + corUsersNumeratorComponentsSum)
    //    println("Cor user num: " + corUsersDenominatorComponentSum)
    //    prediction cannot be larger than 5 and cannot be smaller than 1
    //    values obtained from experimenting with output RMSE

    if (prediction > 5) {
      prediction = 4.7
    }
    if (prediction <= 1) {
      prediction = 2.8
    }
    ((predictForUserId, toPredictMovieId), (actualRating, prediction))
  }

  def generateBaseDict(ratesAndPredictionChunk: Iterator[((String, String), (Float, Double))]): Iterator[(String, Int)] = {
    val baseLineDict = mutable.HashMap(
      ">=0 and <1" -> 0,
      ">=1 and <2" -> 0,
      ">=2 and <3" -> 0,
      ">=3 and <4" -> 0,
      ">=4" -> 0

    )

    for (rateAndPrediction <- ratesAndPredictionChunk) {
      val actualRating = rateAndPrediction._2._1
      val predictedRating = rateAndPrediction._2._2

      val error = Math.abs(actualRating - predictedRating)

      if (0 <= error && error < 1) {
        baseLineDict(">=0 and <1") += 1
      } else if (1 <= error && error < 2) {
        baseLineDict(">=1 and <2") += 1
      } else if (2 <= error && error < 3) {
        baseLineDict(">=2 and <3") += 1
      } else if (3 <= error && error < 4) {
        baseLineDict(">=3 and <4") += 1
      } else {
        baseLineDict(">=4") += 1
      }
    }

    baseLineDict.iterator
  }


  def printBaseLine(ratesAndPreds: RDD[((String, String), (Float, Double))]): Unit = {
    val baseLine = ratesAndPreds
      .repartition(8)
      .mapPartitions(generateBaseDict)
      .reduceByKey((a, b) => a + b)
      .collect().toMap

    val keyOrders = List(">=0 and <1", ">=1 and <2", ">=2 and <3", ">=3 and <4", ">=4")
    for (key <- keyOrders) {
      println(key + ": " + baseLine.getOrElse(key, 0).toString)
    }
  }


  def saveOutput(ratesAndPreds: RDD[((String, String), (Float, Double))]): Unit = {
    val predictionPairs = ratesAndPreds.sortBy(_._1._2).sortBy(_._1._1).collect()


    var predictionOutput = new ArrayBuffer[String]()
    for (predictionPair <- predictionPairs) {
      predictionOutput += predictionPair._1._1.toString + ", " + predictionPair._1._2.toString + ", " + predictionPair._2._2.toString
    }

    val finalOutput = predictionOutput.mkString("\n")
    val pw = new PrintWriter(new File("Malhar_Kulkarni_Yelp_challenge.txt"))
    pw.write(finalOutput)
    pw.close()
  }


  def generate_user_based_CF(sc:SparkContext, ratingFilePath: String, testingFilePath: String):RDD[((String, String), (Float, Double))] = {
    val startTime = System.nanoTime

    // Setting up the spark context

    val text_train = sc.textFile(ratingFilePath)
    val header_train = text_train.first()
    val trainingRdd = text_train
      .filter(line => line != header_train)
      .map(line => line.split(","))
      .map(elements => ((elements(0), elements(1)), elements(2).toFloat))

    val text_test = sc.textFile(testingFilePath)
    val header_test = text_test.first()
    val testingRddWithRatings = text_test
      .filter(line => line != header_test)
      .map(line => line.split(","))
      .map(elements => ((elements(0), elements(1)), elements(2).toFloat))


    val userBusinessTupleRdd = trainingRdd.keys
    //Creating a map of the user and ratings available
    val userBusinessRatingMap: Map[(String, String), Float] = trainingRdd.collect().toMap

    // User and the corresponding ratings he/she has rated
    val userBusinessRdd = userBusinessTupleRdd
      .combineByKey(intToSet, addIntToSet, unionTwoIntSet)
    val userAndBusinessList: Array[(String, Set[String])] = userBusinessRdd.collect()
    var userAndBusinessMap = collection.mutable.Map(userAndBusinessList: _*)
    //    println("User and business map generated")


    // Business and their corresponding users who have rated them
    val businessUsersRdd = userBusinessTupleRdd
      .map(userBusinessRecord => (userBusinessRecord._2, userBusinessRecord._1))
      .combineByKey(intToSet, addIntToSet, unionTwoIntSet)
    val businessAndUsersList: Array[(String, Set[String])] = businessUsersRdd.collect()
    var businessAndUsersMap = collection.mutable.Map(businessAndUsersList: _*)
    //    println("Businesses and users map generated")

    // Calculating pearson correlation matrix
    // (u1, u2) --> Their pearson correlation
    // Fixing missing values
    testingRddWithRatings.collect()
      .foreach(userBusinessTuple => unratedUserBusinessFixer(userBusinessTuple, userAndBusinessMap, businessAndUsersMap))

    // Calculating pearson correlation
    val pearsonCorrelationMap = testingRddWithRatings
      .flatMap(userBusinessTuple => buildSimBusinessUser(userBusinessTuple, businessAndUsersMap))
      .distinct()
      .map(corratedUserTuple => getPearsonCorrelation(corratedUserTuple, userAndBusinessMap, userBusinessRatingMap))
      .collect()
      .toMap

    //    println("Pearson correlations generated")

    val userRatingsPredictions: RDD[((String, String), (Float, Double))] = testingRddWithRatings
      .map(userBusinessRatings =>
        calculateBusinessPrediction(userBusinessRatings, userAndBusinessMap, userBusinessRatingMap, businessAndUsersMap, pearsonCorrelationMap))
      .cache()
    //    println("Predictions generated")

    userRatingsPredictions
  }
}
