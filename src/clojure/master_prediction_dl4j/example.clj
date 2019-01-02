(ns master-prediction-dl4j.example
  (:require [master-prediction-dl4j.data :as data]
            [master-prediction-dl4j.prediction :as prediction])
  (:import (javasrc NeuralNet)
           (org.deeplearning4j.nn.multilayer MultiLayerNetwork)))

(def net (MultiLayerNetwork. (NeuralNet/getNetConfiguration)))


(prediction/init-network net)
(prediction/evaluate-mape net data/normalizer data/test-data)
(prediction/net-predict net data/normalizer (data/prepare-input-vector data/input-test data/normalizer))
(prediction/train-network net data/train-data data/test-data data/normalizer)

(prediction/save-network net "dl4j_nn122t")

(def net2 (prediction/load-network "dl4j_nn140"))
(prediction/evaluate-mape net2 data/normalizer data/test-data)
(prediction/net-predict net2 data/normalizer (data/prepare-input-vector data/input-test data/normalizer))
(prediction/train-network net2 data/train-data data/test-data data/normalizer)



