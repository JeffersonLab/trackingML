package org.jlab.ml.tracking.nn;

import java.io.IOException;
import java.util.List;
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
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.optimize.listeners.TimeIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;

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
    
    private int nEpochs = 20;
    
    public static void main(String[] args){
        
        System.out.println("running ML tracking learning algorithm...");
        
        TrackingModel model = new TrackingModel();
        
        MultiLayerConfiguration configuration = model.getConfiguration();
                
        MultiLayerNetwork network = new MultiLayerNetwork(configuration);        
        network.init();
        
        network.addListeners(new ScoreIterationListener(1));        
        network.addListeners(new TimeIterationListener(1));

        LineImageDataIterator iter = new LineImageDataIterator();

        iter.readDirectory("mldata");
        INDArray  aInputs = iter.getInputArray();
        INDArray aOutputs = iter.getOutputArray();
        
        
        List<String> names = network.getLayerNames();
        
        for(int i = 0; i < names.size(); i++){
            System.out.println(String.format("%3d : %s", i, names.get(i)));
        }
        
        for(int i = 0; i < 1000; i++){
            System.out.println("running epoch # " + i);
            network.fit(aInputs, aOutputs);
            double score = network.score();
            System.out.println("epoch # " + i + " is complete with score = " + score);
        }
        
        
        
        
        
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
