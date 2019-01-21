package example

import botkop.numsca.Tensor
import botkop.{numsca => np}

object NeuralNetwork{

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
      case Nil => List(prev_delta)
      case weight :: Nil => List(prev_delta)
      case weight :: reversed_weitghs_tail => {
        val error = prev_delta.dot(weight.T)
        val delta = error * derivate(activations.head)
        backpropagate(reversed_weitghs_tail, activations.tail, delta) :+ prev_delta
      }
    }



    def learnEpoch(weights: List[Tensor], dataset: List[(Tensor, Tensor)], alpha: Double): List[Tensor] = dataset match {
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
        learnEpoch(updated_weights, dataset.tail, alpha)
      }
    }

    def loss(weights: List[Tensor], predictions: Tensor)(implicit label_set: Tensor): Double = weights match {
      case Nil => 0.5 * np.sum((predictions - label_set) ** 2)
      case weight :: weights_tail =>
        loss(weights_tail, 1/(1 + np.exp(-np.dot(predictions, weight))))
    }

    def predict(weights: List[Tensor], prediction: Tensor): Tensor = weights match {
      case Nil => prediction
      case weight :: weights_tail =>
          predict(weights_tail, 1 /( 1 + np.exp(-np.dot(prediction, weight))))
    }


}

