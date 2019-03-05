# Implementation of neural networks for predicting the consumption of electricity using the programming language of Clojure

### Master thesis

## The second implementation - source code

## Usage

#### Create new neural network

> ;; we create new neural network 

>> ;; setting parameters and create neural network

>> (set! NeuralNet/momentum 0.9)

>> (set! NeuralNet/learning_rate 1.5557E-3)

>> (def net (MultiLayerNetwork. (NeuralNet/getNetConfiguration)))

>> ;; init network

>> (prediction/init-network net)

#### Train network

>> ;; train network with params: 150 epoch and miniBatchSize is 10

>> (prediction/train-network net data/train-data data/test-data data/normalizer 150 10)

#### Network evaluation

> ;; evaluate neural network by MAPE metric

>> (prediction/evaluate-mape net data/normalizer data/test-data)

#### How to use network

> ;; first, we must prepare input vector and temp variables

>> (def input-test [2010 9 25 7 0 3424 3060 2861 2772 2761 2971 3435 4015 4195 4261 4215
                 4268 4225 4161 4047 4003 3995 3992 4365 4956 4849 4670 4273 3848 2761
                 4956 93622 14 19.5 25 15.96 14 19.27 25 15.82 23.27 48.95 1011.73 3452
                 3104 2886 2794 2757 2890 3084 3617 4017 28601 17 18.59090909 22 17.80976667
                 17 18.59090909 22 17.62029836 28.68181818 68 1002.318182])

> ;; then, we can do predict
>> (prediction/predict net data/normalizer prep-input)

#### Save state in file

> ;; when your network good trained, you can save state in file.

> (prediction/save-network net "dl4j_nn")

#### Load network from file

> ;; create network from file with filename "nn-net.csv"

>> (def new-net (prediction/load-network "dl4j_nn140"))


## License

Copyright © 2019 Vladimir Urosevic

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
