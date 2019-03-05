(ns ^{:author "Vladimir Urosevic"}
  master-prediction-dl4j.example
  (:require [master-prediction-dl4j.data :as data]
            [master-prediction-dl4j.prediction :as prediction]
            [criterium.core :refer :all])
  (:import (javasrc NeuralNet)
           (org.deeplearning4j.nn.multilayer MultiLayerNetwork)
           (org.deeplearning4j.datasets.iterator.impl ListDataSetIterator)))

;; setting parameters and create neural network
(set! NeuralNet/momentum 0.9)
(set! NeuralNet/learning_rate 1.5557E-3)
(def net (MultiLayerNetwork. (NeuralNet/getNetConfiguration)))

;; init network
(prediction/init-network net)

;; evaluate neural network by MAPE metric
(prediction/evaluate-mape net data/normalizer data/test-data)

;; predict by specified input vector
(prediction/predict net data/normalizer (data/prepare-input-vector data/input-test data/normalizer))

;; prepare file for drawing diagram of convergence
(prediction/create-predict-file net data/normalizer data/test-data "predict-dl4j-10-100.csv")

;; training benchmark
(criterium.core/with-progress-reporting
(criterium.core/quick-bench
  (prediction/train-network net data/train-data data/test-data data/normalizer 1 1)
  )
)

;; train network with params: 150 epoch and miniBatchSize is 10
(prediction/train-network net data/train-data data/test-data data/normalizer 150 10)


;; loading and saving neural network
(prediction/save-network net "dl4j_nn")
(def net2 (prediction/load-network "dl4j_nn140"))


(prediction/evaluate-mape net2 data/normalizer data/test-data)
(prediction/predict net2 data/normalizer (data/prepare-input-vector data/input-test data/normalizer))
(prediction/train-network net2 data/train-data data/test-data data/normalizer 150 15)


;; standard evaluation by deeplearning4j
(prediction/evaluate net (ListDataSetIterator. (.asList data/train-data)))

;; preparing specified input vector
(def prep-input (data/prepare-input-vector data/input-test data/normalizer))

(prediction/predict net data/normalizer prep-input)

(criterium.core/with-progress-reporting
(criterium.core/quick-bench
  (prediction/predict net data/normalizer prep-input)
  )
 )

