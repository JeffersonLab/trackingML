/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.jlab.ml.tracking.nn;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;



/**
 *
 * @author gavalian
 */
public class TrackingModel {
 
    
    

    private int[] inputShape = new int[] {3, 224, 224};
    
    public MultiLayerConfiguration getConfiguration(){
        
        ConvolutionLayer layer0 = new ConvolutionLayer.Builder(3,3)
        .nIn(1)
        .nOut(2)
        .stride(1,1)
        .padding(2,2)
        .weightInit(WeightInit.XAVIER)
        .name("First convolution layer")
        .activation(Activation.RELU)
        .build();
        
        ConvolutionLayer layer1 = new ConvolutionLayer.Builder(3,3)
        .nIn(2)
        .nOut(1)
        .stride(1,1)
        .padding(2,2)
        .weightInit(WeightInit.XAVIER)
        .name("Second convolution layer")
        .activation(Activation.RELU)
        .build();
        
        SubsamplingLayer layer2pool = new SubsamplingLayer.Builder()
                .poolingType(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2,2).stride(2,2).build();
        
        OutputLayer   layerOut = new OutputLayer.Builder()
                .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nOut(180)
                .activation(Activation.SOFTMAX).build();
        
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
        .seed(12345)
        .iterations(1)
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .learningRate(0.01)
        .regularization(true)
        .l2(0.0004)
        .updater(Updater.NESTEROVS)
        .momentum(0.9)
        .list()
            .layer(0, layer0).layer(1, layer1).layer(2, layer2pool).layer(3, layerOut)
        .pretrain(false)
        .backprop(true)
        .setInputType(InputType.convolutional(200,200,1))
        .build();
        /*MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(45).layer( new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nIn(inputShape[0]).nOut(64)
                                .cudnnAlgoMode(cudnnAlgoMode).build())
                .list().build();*/
        return configuration;
    }
    
}
