import java.io.{File, PrintWriter}

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer, ListBuffer}

object Malhar_Kulkarni_UserBasedCF {

  // --------------------------------------------------
  // Utility functions required for the combiner
  // --------------------------------------------------

  //RMSE to beat 1.0523

  def intToSet(x: String): Set[String] = Set(x)

  def addIntToSet(acc: Set[String], x: String): Set[String] = acc.union(Set(x))

  def unionTwoIntSet(acc1: Set[String], acc2: Set[String]): Set[String] = acc1.union(acc2)


  // --------------------------------------------------
  // Functions to help data management and generation
  // of Pearson coeff
  // --------------------------------------------------
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

    val user1BusinessRatings = commonBusinesses.map(BusinessId => userBusinessRatingsMap((user1, BusinessId))).toList
    val user2BusinessRatings = commonBusinesses.map(BusinessId => userBusinessRatingsMap((user2, BusinessId))).toList

    ((user1, user2), pearsonCorrelation(user1BusinessRatings, user2BusinessRatings))
  }


  def pearsonCorrelation(user1BusinessRatings: List[Float], user2BusinessRatings: List[Float], log: Boolean = false): Double = {
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


    val predictForUserAverageRating = if (userRatingList.isEmpty) 3.0 else userRatingList.sum / userRatingList.size

    // Get the list of users who have rated movie_id
    if (!businessAndUsersMap.contains(toPredictMovieId)) {
      return ((predictForUserId, toPredictMovieId), (actualRating, predictForUserAverageRating))
    }

    val corUserList = businessAndUsersMap.getOrElse(toPredictMovieId, Set.empty)

    // Get the pearson correlation values of these users with user_id
    var corUsersNumeratorComponents: List[Float] = List.empty
    var corUsersDenominatorComponents: List[Float] = List.empty

    for (corUser <- corUserList) {
      // calculating correlated user's average rating
      val moviesRatedByCorUsers = userAndBusinessMap.getOrElse(corUser, Set.empty)
      val corUserRatingList = moviesRatedByCorUsers
        .toList
        .filter(userMovieId => userMovieId != toPredictMovieId)
        .map(userMovieId => userBusinessRatingMap((corUser, userMovieId)))
      val corUserAvgRating = corUserRatingList.sum / corUserRatingList.size

      // Fetching pearson correlation of the correlated user
      if (pearsonCorrelationMap.contains((predictForUserId, corUser))) {
        val pearsonCorForCorUser = pearsonCorrelationMap((predictForUserId, corUser))
        if (pearsonCorForCorUser > 0) {
          val corUserNumeratorComponent = (userBusinessRatingMap((corUser, toPredictMovieId)) - corUserAvgRating) * pearsonCorForCorUser
          corUsersDenominatorComponents :+ Math.abs(pearsonCorForCorUser)
          corUsersNumeratorComponents :+ corUserNumeratorComponent
        }
      }
    }

    val corUsersNumeratorComponentsSum: Float = corUsersNumeratorComponents.sum
    val corUsersDenominatorComponentSum: Float = corUsersDenominatorComponents.sum

    var prediction: Double = 0.0
    if (corUsersNumeratorComponentsSum == 0 || corUsersDenominatorComponentSum == 0) {
      prediction = predictForUserAverageRating
    } else {
      prediction = predictForUserAverageRating + (corUsersNumeratorComponentsSum / corUsersDenominatorComponentSum)
    }

    if (prediction > 5) {
      prediction = 5.0
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
    val pw = new PrintWriter(new File("Malhar_Kulkarni_UserBasedCF.txt"))
    pw.write(finalOutput)
    pw.close()
  }


  def main(args: Array[String]): Unit = {
    val startTime = System.nanoTime

    val ratingFilePath = args(0)
    val testingFilePath = args(1)

    // Setting up the spark context
    val sparkConf = new SparkConf().setAppName("UserBasedCollaborativeFiltering").setMaster("local[*]")
    val sc = new SparkContext(sparkConf)
    sc.setLogLevel("WARN")

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

    val userRatingsPredictions = testingRddWithRatings
      .map(userBusinessRatings =>
        calculateBusinessPrediction(userBusinessRatings, userAndBusinessMap, userBusinessRatingMap, businessAndUsersMap, pearsonCorrelationMap))
      .cache()
    //    println("Predictions generated")

    val MSE = userRatingsPredictions.map(userPred => (userPred._2._1 - userPred._2._2) * (userPred._2._1 - userPred._2._2)).sum / userRatingsPredictions.count()

    //Printing the baseline
    printBaseLine(userRatingsPredictions)
    println("RMSE: " + Math.sqrt(MSE).toString)
    saveOutput(userRatingsPredictions)
    println("Time: " + (System.nanoTime() - startTime) / 1e9d + " sec") // Saving the output in text file
  }
}
