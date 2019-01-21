package example

import botkop.numsca.Tensor
import botkop.{numsca => np}
import com.typesafe.scalalogging.LazyLogging

import scala.annotation.tailrec

object NeuralNetwork extends LazyLogging{

  def main(args: Array[String]): Unit = {

    import Training._
    val network = List(2, 2, 1)
    val alpha = 0.5
    val epochs = 3000
    val weights: List[Tensor] = initializeWeights(network)

    val derivate: Tensor => Tensor = (activation: Tensor) => activation * (1 - activation)

    def activate(weights: List[Tensor], activation: Tensor): List[Tensor] = weights match {
      case Nil => List(activation)
      case weight :: weights_tail => {
        val net: Tensor = np.dot(activation, weight)
        val out = 1 / (1 + np.exp(-net))
        activate(weights_tail, out) :+ activation
      }
    }

    def backpropagate(reversed_weights: List[Tensor], activations: List[Tensor], prev_delta: Tensor): List[Tensor] = reversed_weights match {
      case weight :: Nil => List(prev_delta)
      case weight :: reversed_weitghs_tail => {
        val error = prev_delta.dot(weight.T)
        val delta = error * derivate(activations.head)
        backpropagate(reversed_weitghs_tail, activations.tail, delta) :+ prev_delta
      }
    }

    val dataset: List[(Tensor, Tensor)] = (for (i <- 0 to biasedTrainSet.shape(0) - 1) yield {
      (biasedTrainSet.slice(i), labelSet.slice(i))
    }).toList


    def learnEpoch(weights: List[Tensor], dataset: List[(Tensor, Tensor)]): List[Tensor] = dataset match {
      case Nil => weights
      case (datapoint, target) :: dataset_tail => {

        // FEEDFORWARD
        val activations = activate(weights, datapoint)

        // BACKPROPAGATION
        val error =  activations.head - target
        val delta = error * derivate(activations.head)
        val deltas = backpropagate(weights.reverse, activations.tail, delta)

        // WEIGHT UPDATE
        val updated_weights = (weights zip (activations.reverse zip deltas))
          .map{_ match {
            case (weight: Tensor, (activation: Tensor, delta: Tensor)) => {
              weight + -alpha * activation.T.dot(delta)
              }
            }
          }
        learnEpoch(updated_weights, dataset.tail)
      }
    }

    def calculate_loss(weights: List[Tensor]) = {

      var predictions = biasedTrainSet
      for (w <- weights) {
        predictions = 1/(1 + np.exp(-np.dot(predictions, w)))
      }

      val loss = 0.5 * np.sum((predictions - labelSet) ** 2)
      loss

    }



    def learn(weights: List[Tensor], epoch: Int=0): List[Tensor] = epoch match {
      case epoch if epoch >= epochs => weights
      case epoch => {
        logger.info(s"Loss ${calculate_loss(weights)} on epoch ${epoch}")
        learn(learnEpoch(weights, dataset), epoch + 1)
      }
    }

    val learnedWeights = learn(weights)


    println("WEIGHTS", weights)
    println("LEARNED WEIGHTS", learnedWeights)

    println(predict(learnedWeights, dataset(0)._1), "dataset", dataset(0))
    println(predict(learnedWeights, dataset(1)._1), "dataset", dataset(1))
    println(predict(learnedWeights, dataset(2)._1), "dataset", dataset(2))
    println(predict(learnedWeights, dataset(3)._1), "dataset", dataset(3))

  }


  def predict(weights: List[Tensor], prediction: Tensor): Tensor = weights match {
    case Nil => prediction
    case weight :: weights_tail =>
        predict(weights_tail, 1 /( 1 + np.exp(-np.dot(prediction, weight))))
  }


}

object Training {

  val trainSet: Tensor = np.array(
    1, 1,
    1, 0,
    0, 1,
    0, 0,
  ).reshape(4,2)

  val labelSet: Tensor = np.array(
    0,
    1,
    1,
    0).reshape(4,1)

  val biasedTrainSet = np.concatenate(Seq(trainSet, np.ones(trainSet.shape(0)).T), 1)

  def initializeWeights(layers: List[Int], w: List[Tensor] = List()): List[Tensor] = {

    if(layers.length < 2) return w
    if(layers.length == 2) w :+ geneateRandomWeight(layers.head + 1, layers.last)
    else initializeWeights(layers.tail, w :+ geneateRandomWeight(layers.head + 1, layers.tail.head + 1))


  }

  def geneateRandomWeight(shape: Int*): Tensor = 2 * np.rand(shape.toArray) - 1
}
