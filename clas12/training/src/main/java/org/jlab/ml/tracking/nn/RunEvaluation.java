/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.jlab.ml.tracking.nn;

import java.io.File;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;
import org.nd4j.linalg.api.ndarray.INDArray;
/**
 *
 * @author gavalian
 */
public class RunEvaluation {
    
    public static double evaluate(List<Double> bins, List<Double> weights){
        double  w = 0.0;
        double ew = 0.0;
        double eb = 0.0;
        for(int i = 0; i < bins.size(); i++){
            w += weights.get(i);
            ew += bins.get(i)*weights.get(i);
            eb += bins.get(i);
        }
        //System.out.println("size = " + bins.size() + " w = " + ew + "  ws = " + w);
        return ew/w;
    }
    
    public static void main(String[] args){
        File file = new File("cnn_str8.nnet");
        LineImageDataIterator iter = new LineImageDataIterator();
        DataProducer        producer = new DataProducer();
        int  nIterations = 500;
        
        long totalTime = 0L;
        try {
            MultiLayerNetwork network = ModelSerializer.restoreMultiLayerNetwork(file);

            for(int n = 0; n < nIterations; n++){
                
                double angle = Math.random()*180.0;
                BufferedImage   image = producer.createImage(200, 200, Math.toRadians(angle), 0.1);
                INDArray        input = iter.getData(image);
                INDArray       output = network.output(input);
                long startTime = System.currentTimeMillis();
                for(int pp = 0; pp < 5; pp++){
                    output = network.output(input);
                }
                long endTime = System.currentTimeMillis();
                
                totalTime = endTime - startTime;
                int       maxBin = 0;
                double  maxValue = 0.0;
                
                List<Double> bins = new ArrayList<Double>();
                List<Double> weights = new ArrayList<Double>();
                for(int i = 0; i < 180; i++){
                    double value = output.getDouble(i);
                    if(value>0.00001){
                        bins.add((double) i);
                        weights.add(value);
                    }
                    if(value>maxValue){
                        maxValue = value;
                        maxBin   = i;
                    }
                }
                double weightedValue = RunEvaluation.evaluate(bins, weights);
                //System.out.println(maxValue + " bin = " + maxBin + "  angle = " + angle);
                System.out.println(String.format("%12.4f %12.5f %12.5f %12.5f %12.5f",(double) maxBin, 
                        weightedValue, angle, maxBin-angle, weightedValue - angle));
            }

            System.out.println(String.format("elapsed time = %d time per event = %8.5f", totalTime,((double) totalTime)/nIterations));
            //System.out.println("output = " + output);
        } catch (IOException ex) {
            Logger.getLogger(RunEvaluation.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}
