(ns master-prediction-dl4j.prediction
  (:require [master-prediction-dl4j.data :as data]
            )
  (:import [org.deeplearning4j.nn.conf NeuralNetConfiguration NeuralNetConfiguration$Builder Updater]
           [org.deeplearning4j.nn.conf.layers DenseLayer$Builder OutputLayer$Builder]
           [org.deeplearning4j.nn.api OptimizationAlgorithm]
           [org.deeplearning4j.nn.weights WeightInit]
           [org.nd4j.linalg.lossfunctions LossFunctions$LossFunction]
           [org.deeplearning4j.nn.multilayer MultiLayerNetwork]
           [org.deeplearning4j.optimize.listeners ScoreIterationListener]
           [org.deeplearning4j.eval Evaluation RegressionEvaluation]
           [org.nd4j.linalg.dataset SplitTestAndTrain DataSet]
           [org.nd4j.linalg.dataset.api.preprocessor
            NormalizerStandardize
            DataNormalization NormalizerMinMaxScaler MultiDataNormalization MultiNormalizerMinMaxScaler MultiNormalizerHybrid]
           [org.deeplearning4j.datasets.datavec RecordReaderDataSetIterator RecordReaderMultiDataSetIterator]
           org.deeplearning4j.api.storage.StatsStorage
           org.deeplearning4j.ui.api.UIServer
           org.deeplearning4j.ui.stats.StatsListener
           org.deeplearning4j.ui.storage.InMemoryStatsStorage
           [javasrc NeuralNet]
           [org.deeplearning4j.util ModelSerializer]
           [org.deeplearning4j.datasets.iterator.impl ListDataSetIterator]
           [org.nd4j.linalg.dataset.api.iterator DataSetIterator]
           [org.nd4j.linalg.dataset.api MultiDataSet]))

(def input-num 64)
(def output-num 1)
(def batch-size 128)
(def num-epochs 30)
(def rng-seed 123)

(set! NeuralNet/momentum 0.9)
(set! NeuralNet/learning_rate 1.5E-3)
(def net (MultiLayerNetwork. (NeuralNet/getNetConfiguration)))

(defn set-ui [net]
  (let [ui-server (UIServer/getInstance)
        stats-storage (InMemoryStatsStorage.)]
    (.setListeners net (into-array `(~(StatsListener. stats-storage))))
    (.attach ui-server stats-storage)
    net))

(defn test-net
  ([net test-iterator]
   (let [eval (.evaluateRegression net test-iterator)]
     (println "Evaluated net")
     (println (.stats eval))
     )))

(defn test-net-predict
  ([net normalizer test-data test-iterator]
   (let [test-copy (.copy test-data)
         predicted (.output net test-iterator)
         - (.revertLabels normalizer predicted)
         - (.revertLabels normalizer (.getLabels test-copy))
         pod2 (map #(double %) (.getLabels test-copy))
         pred1 (map #(double %) predicted)
         ev (Evaluation. ) ]

     (println "Predicted ...")
     (map #(vector %1 %2) pred1 pod2)
     ;;(.eval ev (.getLabels test-data))
     )))

(defn train-network []
  (let [- (.reset data/all-data-reader)
        allData-iterator (RecordReaderDataSetIterator.
                           data/all-data-reader 2647 64 64 true)
        allData        (DataSet.)
        allData        (.next allData-iterator)
        testAndTrain   (.splitTestAndTrain allData 0.7)
        train-data     (.getTrain testAndTrain)
        train-data-itr (ListDataSetIterator. (.asList train-data))
        test-data      (.getTest testAndTrain)
        test-copy      (.copy test-data)
        test-data-itr (ListDataSetIterator. (.asList test-data))
        normalizer     (NormalizerMinMaxScaler. 0 1)
        -              (.fitLabel normalizer true)
        -              (.fit normalizer train-data)
        -              (.transform normalizer train-data)
        -              (.transform normalizer test-data)
        ready-for-more true]
    (println "Initializing net...")
    (.init net)
    (println "Initialized net")

    (set-ui net)
    (println "UI set")

    (doseq [n (range 0 num-epochs)]
      (.reset train-data-itr)
      (println "test-iterator reset")
      (.fit net train-data-itr))
    (println "Trained net")

    (ModelSerializer/writeModel
      net (java.io.File. "resources/net-8") ready-for-more
      )
    (println "Saved net")

    (.reset test-data-itr)
    (test-net-predict net normalizer test-data test-data-itr)
    ))

(train-network)
;;(map #(reduce + %) [[1 2] [2 3] [3 4]])
