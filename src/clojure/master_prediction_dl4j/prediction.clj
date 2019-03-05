(ns ^{:author "Vladimir Urosevic"}
  master-prediction-dl4j.prediction
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

(defn set-ui [net]
  (let [ui-server (UIServer/getInstance)
        stats-storage (InMemoryStatsStorage.)]
    (.setListeners net (into-array `(~(StatsListener. stats-storage))))
    (.attach ui-server stats-storage)
    net))

(defn evaluate
  ([net test-iterator]
   (let [eval (.evaluateRegression net test-iterator)]
     (println "Evaluated net")
     (println (.stats eval))
     )))

(defn absulute-percent-error [v]
  (let [f (first v)
        s (second v)
        o (- f s)]
    (* 100 (Math/abs (double (/ o s))))))

(defn evaluate-mape
  ([net normalizer test-data]
   (let [test-copy (.copy test-data)
         test-iterator (ListDataSetIterator. (.asList test-data))
         predicted (.output net test-iterator)
         - (.revertLabels normalizer predicted)
         - (.revertLabels normalizer (.getLabels test-copy))
         pod (map #(double %) (.getLabels test-copy))
         pred (map #(double %) predicted)
         pom (vec (map #(vector %1 %2) pred pod))
         ev (Evaluation. ) ]

     (let [ape (map #(absulute-percent-error %) pom)
           uku (count ape)
           sumape (apply + (map double ape))]
       (/ sumape uku))
    )))

(defn create-predict-file
  ([net normalizer test-data]
   (let [test-copy (.copy test-data)
         test-iterator (ListDataSetIterator. (.asList test-data))
         predicted (.output net test-iterator)
         - (.revertLabels normalizer predicted)
         - (.revertLabels normalizer (.getLabels test-copy))
         pod (map #(double %) (.getLabels test-copy))
         pred (map #(double %) predicted)
         pom (vec (map #(vector %1 %2) pred pod))
         ev (Evaluation.)]

      (doseq [x (range (count pod))]
            (data/write-file "predict-dl4j-10-100200200100.csv" (str x "," (nth pod x) "," (nth pred x) "\n"))
          )
     )))

(defn predict
  ([net normalizer test-data]
   (let [test-iterator (ListDataSetIterator. (.asList test-data))
         predicted (.output net test-iterator)
         - (.revertLabels normalizer predicted)
         pred (map #(double %) predicted)]
     pred)))

(defn init-network
  [net]
  (println "Initializing net...")
  (.init net)
  (println "Initialized net")
  (set-ui net)
  (println "UI set"))

(defn train-network [net train-data test-data normalizer num-epochs mini-batch-size]
  (let [train-data-itr (ListDataSetIterator. (.asList train-data) mini-batch-size)]

    (doseq [n (range 0 num-epochs)]
      (.reset train-data-itr)
      (.fit net train-data-itr)
      (println (str n " , " (evaluate-mape net normalizer test-data)))
      ;;(data/write-file "konvergencijadl4j_minibatch_test.csv" (str n "," (evaluate-mape net normalizer test-data) "\n"))
      )

    (println "Trained net")
    (evaluate-mape net normalizer test-data)
    ))

(defn save-network
  [net filename]
  (ModelSerializer/writeModel
    net (java.io.File. (str "resources/" filename)) true)
  )

(defn load-network
  [filename]
  (ModelSerializer/restoreMultiLayerNetwork (str "resources/" filename))
  )


