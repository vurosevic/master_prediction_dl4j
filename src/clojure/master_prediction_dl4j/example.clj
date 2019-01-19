(ns master-prediction-dl4j.example
  (:require [master-prediction-dl4j.data :as data]
            [master-prediction-dl4j.prediction :as prediction])
  (:import (javasrc NeuralNet)
           (org.deeplearning4j.nn.multilayer MultiLayerNetwork)
           (org.deeplearning4j.datasets.iterator.impl ListDataSetIterator)))

(set! NeuralNet/momentum 0.9)
(set! NeuralNet/learning_rate 1.5E-3)
(def net (MultiLayerNetwork. (NeuralNet/getNetConfiguration)))


(prediction/init-network net)
(prediction/evaluate-mape net data/normalizer data/test-data)
(prediction/predict net data/normalizer (data/prepare-input-vector data/input-test data/normalizer))
(prediction/train-network net data/train-data data/test-data data/normalizer 25)

(prediction/save-network net "dl4j_nn122t")

(def net2 (prediction/load-network "dl4j_nn140"))
(prediction/evaluate-mape net2 data/normalizer data/test-data)
(prediction/predict net2 data/normalizer (data/prepare-input-vector data/input-test data/normalizer))
(prediction/train-network net2 data/train-data data/test-data data/normalizer 15)


(prediction/evaluate net (ListDataSetIterator. (.asList data/train-data)))

