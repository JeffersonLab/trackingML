package org.jlab.ml.tracking.nn;

import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author gavalian
 */
public class RunTraining {
    
    
    public static void main(String[] args){
        
        System.out.println("running ML tracking learning algorithm...");
        
        ConvolutionLayer layer0 = new ConvolutionLayer.Builder(5,5)
        .nIn(3)
        .nOut(16)
        .stride(1,1)
        .padding(2,2)
        .weightInit(WeightInit.XAVIER)
        .name("First convolution layer")
        .activation(Activation.RELU)
        .build();
        
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
        .seed(12345)
        .iterations(1)
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .learningRate(0.001)
        .regularization(true)
        .l2(0.0004)
        .updater(Updater.NESTEROVS)
        .momentum(0.9)
        .list()
            .layer(0, layer0)
        .pretrain(false)
        .backprop(true)
        .setInputType(InputType.convolutional(32,32,3))
        .build();
        
        MultiLayerNetwork network = new MultiLayerNetwork(configuration);
        network.init();
        
        try {
            
            ModelSerializer.writeModel(network, "data_file.nnet", true);
            
            
            /*
            TrackingModel model = new TrackingModel();
            MultiLayerConfiguration   config = model.getConfiguration();
            MultiLayerNetwork        network = new MultiLayerNetwork(config);
            
            List<String> layerNames = network.getLayerNames();
            for(int i = 0; i < layerNames.size(); i++){
            System.out.println(String.format("%3d : %12s", i, layerNames.get(i)));
            }*/
            //ZooModel zooModel = ;
        } catch (IOException ex) {
            Logger.getLogger(RunTraining.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}
