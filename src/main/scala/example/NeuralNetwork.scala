package example

import botkop.numsca.Tensor
import botkop.{numsca => np}

object NeuralNetwork{

    val sigmoid_activation: Tensor => Tensor = (net: Tensor) =>  1 / (1 + np.exp(-net))
    val sigmoid_derivate: Tensor => Tensor = (activation: Tensor) => activation * (1 - activation)


  /**
    *
    * @param weights
    * @param activation
    * @return weights
    */
    def forward(weights: List[Tensor], prediction: Tensor): List[Tensor] = weights match {
      case Nil => List(prediction)
      case weight :: weights_tail => {
        val net: Tensor = np.dot(prediction, weight)
        val activation: Tensor = sigmoid_activation(net)
        forward(weights_tail, activation) :+ prediction
      }
    }

  /**
    * Calculate an impact of every layer's weight we have in place on a missed target
    *
    * @param reversed_weights
    * @param predictions
    * @param prev_delta
    * @return deltas
    */
    def backpropagate(reversed_weights: List[Tensor], predictions: List[Tensor], prev_delta: Tensor): List[Tensor] = reversed_weights match {
      // in case there were no hidden layers provided
      case Nil => List(prev_delta)
      // first applied wight could not have any prediction before, so return its previous error delta
      case weight :: Nil => List(prev_delta)
      // find an error's magnitude for prediction we made with this specific weight
      case weight :: reversed_weitghs_tail => {
        //  calculate a scalar product of a weight applied and an error magnitude this weight created
        val error = prev_delta.dot(weight.T)
        // put it in relation with the magnitude of a prediction, that was done/activated in this layer
        val delta = error * sigmoid_derivate(predictions.head)
        // put it on stack in the following rucursive call
        backpropagate(reversed_weitghs_tail, predictions.tail, delta) :+ prev_delta
      }
    }

  /**
    * Entry point to manage a calculation of weights which used to make a prediction on a dataset
    *
    * @param weights
    * @param rules
    * @param alpha
    * @return weights
    */
    def learnEpoch(weights: List[Tensor], rules: List[(Tensor, Tensor)], alpha: Double): List[Tensor] = rules match {
      case Nil => weights
      case (rule_input, rule_output) :: rules_tail => {

        // forward pass
        val predictions_stack = forward(weights, rule_input)

        // backward pass
        val error_deltas = predictions_stack match {
          case last_prediction :: prev_predictions => {
            // prepare the last error based on rule's output/target/label
            val last_error = last_prediction - rule_output
            // put it in ralation to prediction's derivate
            val last_error_delta = last_error * sigmoid_derivate(last_prediction)
            // go through the rest of predictions we made before
            backpropagate(weights.reverse, prev_predictions, last_error_delta)
          }
          case _ => Nil
        }

        // update weigths by gradient decent
        val updated_weights = (weights, predictions_stack.reverse, error_deltas).zipped.toList.map {
          case (weight: Tensor, activation: Tensor, error_delta: Tensor) =>
            // add gradient descent
            weight + -alpha * activation.T.dot(error_delta)
        }
        learnEpoch(updated_weights, rules_tail, alpha)
      }
    }


    ////////////////////////
    // Helper function
    ///////////////////////

  /**
    * Calculate an index of missing a target
    *
    * @param weights
    * @param predictions
    * @param label_set
    * @return error magnitude
    */
    def loss(weights: List[Tensor], predictions: Tensor)(implicit rules_output: Tensor): Double = weights match {
      case Nil => np.sum((predictions - rules_output) ** 2)/2
      case weight :: weights_tail =>
        loss(weights_tail, sigmoid_activation(np.dot(predictions, weight)))
    }

  /**
    * do a prediction on dataset
    *
    * @param weights
    * @param prediction
    * @return prediction index
    */
    def predict(weights: List[Tensor], prediction: Tensor): Tensor = weights match {
      case Nil => prediction
      case weight :: weights_tail =>
          predict(weights_tail, sigmoid_activation(np.dot(prediction, weight)))
    }

}

