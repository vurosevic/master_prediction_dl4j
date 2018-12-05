(defproject master_prediction_dl4j "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :jvm-opts ["-Xms1024m" "-Xmx8000m"]
  :source-paths ["src/clojure"]
  :java-source-paths ["src/java"]
  :dependencies [[org.clojure/clojure "1.9.0"]
                 [org.clojure/tools.logging "0.3.1"]
                 [org.clojure/data.csv "0.1.4"]
                 [ch.qos.logback/logback-classic "1.2.3"]
                 [mount "0.1.11"]
                 [org.datavec/datavec-api "1.0.0-beta2"]
                 [org.deeplearning4j/deeplearning4j-nlp "1.0.0-beta2"]
                 [org.deeplearning4j/deeplearning4j-core "1.0.0-beta2"]
                 [org.deeplearning4j/deeplearning4j-nn "1.0.0-beta2"]
                 [org.deeplearning4j/deeplearning4j-ui_2.11 "1.0.0-beta2"]
                 [org.nd4j/nd4j-common "1.0.0-beta2"]
                 [org.nd4j/nd4j-native-platform "1.0.0-beta2"]
                 [org.slf4j/slf4j-log4j12 "1.7.1"]
                 [log4j/log4j "1.2.17" :exclusions [javax.mail/mail
                                                    javax.jms/jms
                                                    com.sun.jmdk/jmxtools
                                                    com.sun.jmx/jmxri]]])
