/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.jlab.ml.tracking.clas12;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.jlab.ml.tracking.nn.RunEvaluation;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author gavalian
 */
public class RunClas12Evaluation {
    public static void main(String[] args){
        // File file = new File("clas12_tracking_RELU.nnet");
       File file = new File("data_file.nnet");
        int  nIterations = 500;
        
        
        long totalTime = 0L;
        try {
            MultiLayerNetwork network = ModelSerializer.restoreMultiLayerNetwork(file);
            
            String filename = "/Users/gavalian/Work/Software/project-6a.0.0/clas_004013_ML.hipo";
            
            //String filename = "/Users/gavalian/Work/Software/project-6a.0.0/data/ML/clas_004148.evio.99.recon.hipo";
            
            Clas12DataLoader loader = new Clas12DataLoader();
            loader.readFile(filename);
            
            List<TrackData>  trackData = loader.tracks();
            List<TrackData>  tracks    = new ArrayList<TrackData>();
            
            for(int i = 0; i < trackData.size(); i++){
                tracks.clear();
                tracks.add(trackData.get(i));
                INDArray input  = loader.getInputArray(tracks);
                INDArray output = network.output(input);
                
                double momentum = output.getDouble(0,0);
                //double theta    = output.getDouble(0,1);
                //double phi      = output.getDouble(0,2);
                
                TrackData data = tracks.get(0);
                System.out.println(String.format("%8.5f %8.5f %8.5f %8.5f",
                        data.parameters()[0],data.parameters()[1],data.parameters()[2],
                        momentum));//,theta,phi ));
            }
            
            } catch (IOException ex) {
            Logger.getLogger(RunEvaluation.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}
