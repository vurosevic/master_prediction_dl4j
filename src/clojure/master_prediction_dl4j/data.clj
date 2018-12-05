(ns master-prediction-dl4j.data
  (:require [clojure.data.csv :as csv]
            [clojure.java.io :as io])
  (:import [org.datavec.api.records.reader.impl.csv CSVRecordReader]
           [org.datavec.api.split FileSplit NumberedFileInputSplit]
           [org.datavec.api.util ClassPathResource]
           [org.deeplearning4j.datasets.datavec RecordReaderDataSetIterator]))


(def data-file-name "train_data.csv")

(let [all-data-reader (CSVRecordReader.)
      _ (.initialize all-data-reader (FileSplit.
                                       (-> (ClassPathResource. data-file-name)
                                           (.getFile))
                                       ))]
  (def all-data-reader all-data-reader))

