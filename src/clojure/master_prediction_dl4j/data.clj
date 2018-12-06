(ns master-prediction-dl4j.data
  (:require [clojure.data.csv :as csv]
            [clojure.java.io :as io])
  (:import [org.datavec.api.records.reader.impl.csv CSVRecordReader]
           [org.datavec.api.split FileSplit NumberedFileInputSplit]
           [org.datavec.api.util ClassPathResource]
           [org.deeplearning4j.datasets.datavec RecordReaderDataSetIterator]
           [org.nd4j.linalg.dataset SplitTestAndTrain DataSet]
           [org.nd4j.linalg.dataset.api.preprocessor
           NormalizerStandardize
           DataNormalization NormalizerMinMaxScaler MultiDataNormalization MultiNormalizerMinMaxScaler MultiNormalizerHybrid]
           [org.deeplearning4j.datasets.datavec RecordReaderDataSetIterator RecordReaderMultiDataSetIterator]
           [org.deeplearning4j.util ModelSerializer]
           [org.deeplearning4j.datasets.iterator.impl ListDataSetIterator]
           [org.nd4j.linalg.dataset.api.iterator DataSetIterator]
           [org.nd4j.linalg.dataset.api MultiDataSet]
           ))


(def data-file-name "train_data.csv")

(let [all-data-reader (CSVRecordReader.)
      _ (.initialize all-data-reader (FileSplit.
                                       (-> (ClassPathResource. data-file-name)
                                           (.getFile))
                                       ))]
(def all-data-reader all-data-reader))

(def allData-iterator (RecordReaderDataSetIterator.
                        all-data-reader 2647 64 64 true))
(def allData        (DataSet.))
(def allData        (.next allData-iterator))
(def testAndTrain   (.splitTestAndTrain allData 0.7))
(def train-data     (.getTrain testAndTrain))
(def test-data      (.getTest testAndTrain))
(def normalizer     (NormalizerMinMaxScaler. 0 1))
(.fitLabel normalizer true)
(.fit normalizer train-data)
(.transform normalizer train-data)
(.transform normalizer test-data)