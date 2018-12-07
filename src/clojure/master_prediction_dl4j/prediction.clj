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
(def num-epochs 10)
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

(defn absulute-p-error [v]
  (let [f (first v)
        s (second v)
        o (- f s)]
    (* 100 (Math/abs (double (/ o s))))))

(defn test-net-predict
  ([net normalizer test-data]
   (let [test-copy (.copy test-data)
         test-iterator (ListDataSetIterator. (.asList test-data))
         predicted (.output net test-iterator)
         - (.revertLabels normalizer predicted)
         - (.revertLabels normalizer (.getLabels test-copy))
         pod2 (map #(double %) (.getLabels test-copy))
         pred1 (map #(double %) predicted)
         pom1 (vec (map #(vector %1 %2) pred1 pod2))
         ev (Evaluation. ) ]

     (println "Predicted ...")


     (let [ape (map #(absulute-p-error %) pom1)
           uku (count ape)
           sumape (apply + (map double ape))]
       (/ sumape uku)
       )

    )))

(defn net-predict
  ([net normalizer test-data]
   (let [test-iterator (ListDataSetIterator. (.asList test-data))
         predicted (.output net test-iterator)
         - (.revertLabels normalizer predicted)
         pred1 (map #(double %) predicted)]

     (println "Predicted ...")
      (-> pred1)
     )))



(defn train-network [net train-data test-data normalizer]
  (let [train-data-itr (ListDataSetIterator. (.asList train-data))
        test-copy      (.copy test-data)
        test-data-itr (ListDataSetIterator. (.asList test-data))
        ready-for-more true]
    (println "Initializing net...")
    (.init net)
    (println "Initialized net")

    (set-ui net)
    (println "UI set")

    (doseq [n (range 0 num-epochs)]
      (.reset train-data-itr)
      (println "test-iterator reset")
      (.fit net train-data-itr)

      (println (test-net-predict net normalizer test-data))
      )

    (println "Trained net")

    (ModelSerializer/writeModel
      net (java.io.File. "resources/net-8") ready-for-more
      )
    (println "Saved net")

    (test-net-predict net normalizer test-data)
    ))

(net-predict net data/normalizer data/test-ds)

(train-network net data/train-data data/test-data data/normalizer)

(absulute-p-error [20 15])


