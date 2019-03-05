/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.jlab.ml.tracking.chambers;

import java.io.File;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.jlab.groot.data.GraphErrors;
import org.jlab.groot.data.H1F;
import org.jlab.groot.ui.TCanvas;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author gavalian
 */
public class RunEvaluationOne {
    public static void main(String[] args){
        // File file = new File("clas12_tracking_RELU.nnet");
       File file = new File("network_chambers.nnet");
        int  nIterations = 500;
        
        
        long totalTime = 0L;
        try {
            MultiLayerNetwork network = ModelSerializer.restoreMultiLayerNetwork(file);
            
            System.out.println(network.summary());
            
            System.out.println("\n>>>>>>>>>>>>>>> END OF SUMMARY <<<<<<<<<<<<<");
            TCanvas c1 = new TCanvas("c1",500,500);
            GraphErrors graph = new GraphErrors();
            H1F         histo = new H1F("h1",100,-5.0,5.0);
            
            for(int i = 0; i < nIterations; i++){
                DataLoader loader = new DataLoader();
                loader.generate(1);
                INDArray input = loader.getInputArray();
                INDArray output = network.output(input);
                double angle = loader.getDetectorHits().get(0).getAngle();
                double angleOut = output.getDouble(0, 0)*50.0-25.0;
                System.out.println(String.format("%8.5f %8.5f", angle ,angleOut));
                graph.addPoint(angle, angleOut, 0.0, 0.0);
                histo.fill(angle-angleOut);
            }
            c1.divide(2,1);
            c1.cd(0);
            c1.draw(graph);
            c1.cd(1);
            c1.draw(histo);
            } catch (IOException ex) {
            Logger.getLogger(org.jlab.ml.tracking.nn.RunEvaluation.class.getName()).log(Level.SEVERE, null, ex);
        
            }
        
        
    }
}
