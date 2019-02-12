package org.jlab.ml.tracking.nn;

import java.util.List;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.zoo.ZooModel;

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
        
        TrackingModel model = new TrackingModel();
        MultiLayerConfiguration   config = model.getConfiguration();
        MultiLayerNetwork        network = new MultiLayerNetwork(config);
        
        List<String> layerNames = network.getLayerNames();
        for(int i = 0; i < layerNames.size(); i++){
            System.out.println(String.format("%3d : %12s", i, layerNames.get(i)));
        }
        //ZooModel zooModel = ;
    }
}
