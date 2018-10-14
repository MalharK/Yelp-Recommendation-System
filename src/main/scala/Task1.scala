import org.apache.spark.SparkContext
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.sql.{Row, SparkSession}

import collection.mutable

object Task1 {

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
    if (rating > 5.0)
      return 5.0
    else if (rating < 1.0)
      return 3.0
    rating
  }


  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .master("local")
      .appName("ALSrecommender")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    //    val input_file = args(0)
    //    val output_file = args(1)

    val text_train = spark.sparkContext.textFile("train_review.csv")
    val header = text_train.first()
    val ratings = text_train
      .filter(line => !line.equals(header))
      .map(line => line.split(","))
      .map(attributes => Rating(assign_int_user_id(attributes(0).trim), assign_int_business_id(attributes(1).trim), attributes(2).trim.toDouble))


    // Evaluate the model on rating data
    val text_test = spark.sparkContext.textFile("test_review.csv")
    val test_ratings = text_test
      .filter(line => !line.equals(header))
      .map(line => line.split(","))
      .map(attributes => Rating(assign_int_user_id(attributes(0).trim), assign_int_business_id(attributes(1).trim), attributes(2).trim.toDouble))

    val test_inputs = text_test
      .filter(line => !line.equals(header))
      .map(line => line.split(","))
      .map(attributes => Row(attributes(0).trim, attributes(1).trim))


    // Build the recommendation model using ALS
    val rank = 15 // Can be changed for better performance
    val num_iterations = 20 // Can be changed for better performance
    val lambda_value = 0.3 // Can be changed for better performance
    val blocks = -1 // Can be changed for better performance
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

    val MSE = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
      val err = r1 - r2
      err * err
    }.mean()
    val RMSE = math.sqrt(MSE)
    println(s"Mean Squared Error = $RMSE")

  }
}
