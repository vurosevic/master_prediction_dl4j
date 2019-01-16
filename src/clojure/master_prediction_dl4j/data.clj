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
           (org.nd4j.linalg.factory Nd4j)
           org.nd4j.linalg.api.ndarray.INDArray
           org.nd4j.linalg.cpu.nativecpu.NDArray))


(def data-file-name "train_data.csv")

(let [all-data-reader (CSVRecordReader.)
      _ (.initialize all-data-reader (FileSplit.
                                       (-> (ClassPathResource. data-file-name)
                                           (.getFile))))
      allData-iterator (RecordReaderDataSetIterator.
                         all-data-reader 2647 64 64 true)
      allData (DataSet.)
      allData (.next allData-iterator)
      testAndTrain (.splitTestAndTrain allData 0.7)]
  (do
    (def train-data (.getTrain testAndTrain))
    (def test-data (.getTest testAndTrain)))
  )

;;(def allData-iterator (RecordReaderDataSetIterator.
;;                        all-data-reader 2647 64 64 true))

;;(def allData (DataSet.))
;;(def allData (.next allData-iterator))

;;(def testAndTrain (.splitTestAndTrain allData 0.7))

;;(def train-data (.getTrain testAndTrain))
;;(def test-data (.getTest testAndTrain))
(def normalizer (NormalizerMinMaxScaler. 0 1))
(.fitLabel normalizer true)
(.fit normalizer train-data)
(.transform normalizer train-data)
(.transform normalizer test-data)

;;(-> allData)
;;(-> test-data)


(def input-test [2010 9 25 7 0 3424 3060 2861 2772 2761 2971 3435 4015 4195 4261 4215
                 4268 4225 4161 4047 4003 3995 3992 4365 4956 4849 4670 4273 3848 2761
                 4956 93622 14 19.5 25 15.96 14 19.27 25 15.82 23.27 48.95 1011.73 3452
                 3104 2886 2794 2757 2890 3084 3617 4017 28601 17 18.59090909 22 17.80976667
                 17 18.59090909 22 17.62029836 28.68181818 68 1002.318182])


(defn indarray
  [data]
  (Nd4j/create (double-array data)))

(def test-row (indarray input-test))
(def test-rowo (indarray [0]))

(def test-ds (DataSet. test-row test-rowo))
(.transform normalizer test-ds)

(defn prepare-input-vector
  [input-vector normalizer]
  (let [input-row (indarray input-test)
        output-row (indarray [0])
        output-ds (DataSet. input-row output-row)
        - (.transform normalizer output-ds)]
    output-ds
    ))

(-> test-ds)

(prepare-input-vector input-test normalizer)

