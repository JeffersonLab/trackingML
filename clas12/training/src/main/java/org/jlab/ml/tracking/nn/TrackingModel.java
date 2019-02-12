/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.jlab.ml.tracking.nn;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;

/**
 *
 * @author gavalian
 */
public class TrackingModel {
 
    
    
    private ConvolutionLayer.AlgoMode cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST;
    private int[] inputShape = new int[] {3, 224, 224};
    
    public MultiLayerConfiguration getConfiguration(){
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(45).layer( new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nIn(inputShape[0]).nOut(64)
                                .cudnnAlgoMode(cudnnAlgoMode).build())
                .list().build();
        return conf;
    }
    
}
