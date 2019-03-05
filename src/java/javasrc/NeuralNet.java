package javasrc;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.*;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class NeuralNet {

    public static double momentum = 0.7;
    public static double learning_rate = 0.0015;

    public static MultiLayerConfiguration getNetConfiguration () {
        return new NeuralNetConfiguration.Builder()
                .seed(123L)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(learning_rate, momentum))
                .miniBatch(true)
                .weightInit(WeightInit.XAVIER)
                //.l2(0.0001)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(64).nOut(100)
                        .activation(Activation.TANH).build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS)
                        .activation(Activation.TANH)
                        .nIn(100).nOut(1).build())
                .pretrain(false).backprop(true).build();
    }

    public static MultiLayerConfiguration getNetConfiguration2 () {
        return new NeuralNetConfiguration.Builder()
                .seed(123L)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(learning_rate, momentum))
                .miniBatch(true)
                .weightInit(WeightInit.XAVIER)
                //.l2(0.0001)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(64).nOut(100)
                        .activation(Activation.TANH).build())
                .layer(1, new DenseLayer.Builder().nIn(100).nOut(100)
                        .activation(Activation.TANH).build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS)
                        .activation(Activation.TANH)
                        .nIn(100).nOut(1).build())
                .pretrain(false).backprop(true).build();
    }

    public static MultiLayerConfiguration getNetConfiguration3 () {
        return new NeuralNetConfiguration.Builder()
                .seed(123L)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(learning_rate, momentum))
                .miniBatch(true)
                .weightInit(WeightInit.XAVIER)
                //.l2(0.0001)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(64).nOut(200)
                        .activation(Activation.TANH).build())
                .layer(1, new DenseLayer.Builder().nIn(200).nOut(200)
                        .activation(Activation.TANH).build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS)
                        .activation(Activation.TANH)
                        .nIn(200).nOut(1).build())
                .pretrain(false).backprop(true).build();
    }

    public static MultiLayerConfiguration getNetConfiguration4 () {
        return new NeuralNetConfiguration.Builder()
                .seed(123L)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(learning_rate, momentum))
                .miniBatch(true)
                .weightInit(WeightInit.XAVIER)
                //.l2(0.0001)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(64).nOut(100)
                        .activation(Activation.TANH).build())
                .layer(1, new DenseLayer.Builder().nIn(100).nOut(200)
                        .activation(Activation.TANH).build())
                .layer(2, new DenseLayer.Builder().nIn(200).nOut(100)
                        .activation(Activation.TANH).build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS)
                        .activation(Activation.TANH)
                        .nIn(100).nOut(1).build())
                .pretrain(false).backprop(true).build();
    }


    public static MultiLayerConfiguration getNetConfiguration5 () {
        return new NeuralNetConfiguration.Builder()
                .seed(123L)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(learning_rate, momentum))
                .miniBatch(true)
                .weightInit(WeightInit.XAVIER)
                //.l2(0.0001)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(64).nOut(100)
                        .activation(Activation.TANH).build())
                .layer(1, new DenseLayer.Builder().nIn(100).nOut(200)
                        .activation(Activation.TANH).build())
                .layer(2, new DenseLayer.Builder().nIn(200).nOut(200)
                        .activation(Activation.TANH).build())
                .layer(3, new DenseLayer.Builder().nIn(200).nOut(100)
                        .activation(Activation.TANH).build())
                .layer(4, new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS)
                        .activation(Activation.TANH)
                        .nIn(100).nOut(1).build())
                .pretrain(false).backprop(true).build();
    }


}
