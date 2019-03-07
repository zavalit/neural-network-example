package example

import botkop.numsca.Tensor
import botkop.{numsca => np}
import com.typesafe.scalalogging.LazyLogging
import example.NeuralNetwork.{learnEpoch, loss, predict}

object TrainXOR extends App with LazyLogging {

  val network = List(2, 2, 1)

  val alpha = 0.5

  val epochs = 3000

  val weights: List[Tensor] = initializeWeights(network)

  val trainSet: Tensor = np.array(
    1, 1,
    1, 0,
    0, 1,
    0, 0,
  ).reshape(4,2)

  implicit val labelSet: Tensor = np.array(
    0,
    1,
    1,
    0).reshape(4,1)

  val biasedTrainSet: Tensor = np.concatenate(Seq(trainSet, np.ones(trainSet.shape(0)).T), 1)

  def initializeWeights(layers: List[Int], w: List[Tensor] = List()): List[Tensor] = {

    if(layers.length < 2) return w
    if(layers.length == 2) w :+ geneateRandomWeight(layers.head + 1, layers.last)
    else initializeWeights(layers.tail, w :+ geneateRandomWeight(layers.head + 1, layers.tail.head + 1))

  }

  def geneateRandomWeight(shape: Int*): Tensor = 2 * np.rand(shape.toArray) - 1

  val dataset: List[(Tensor, Tensor)] = (for (i <- 0 to biasedTrainSet.shape(0) - 1) yield {
    (biasedTrainSet.slice(i), labelSet.slice(i))
  }).toList

  def learn(weights: List[Tensor], epoch: Int=0): List[Tensor] = epoch match {
    case epoch if epoch >= epochs => weights
    case epoch => {
      if (epoch % 100 == 0) logger.info(s"Loss ${loss(weights, biasedTrainSet)} on epoch ${epoch}")
      learn(learnEpoch(weights, dataset, alpha), epoch + 1)

    }
  }

  val learnedWeights = learn(weights)


  logger.info(s"INPUT WEIGHTS: ${weights}")
  logger.info(s"LEARNED WEIGHTS: ${learnedWeights}")

  logger.info(s"input: ${dataset(0)._1} => expected output:${dataset(0)._2},  predicted output: ${predict(learnedWeights, dataset(0)._1)}")
  logger.info(s"input: ${dataset(1)._1} => expected output:${dataset(1)._2},  predicted output: ${predict(learnedWeights, dataset(1)._1)}")
  logger.info(s"input: ${dataset(2)._1} => expected output:${dataset(2)._2},  predicted output: ${predict(learnedWeights, dataset(2)._1)}")
  logger.info(s"input: ${dataset(3)._1} => expected output:${dataset(3)._2},  predicted output: ${predict(learnedWeights, dataset(3)._1)}")


}
