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
import org.nd4j.linalg.api.ndarray.INDArray;
/**
 *
 * @author gavalian
 */
public class RunEvaluation {
    public static void main(String[] args){
        File file = new File("cnn_str8.nnet");
        LineImageDataIterator iter = new LineImageDataIterator();
        DataProducer        producer = new DataProducer();
        
        try {
            MultiLayerNetwork network = ModelSerializer.restoreMultiLayerNetwork(file);
            long startTime = System.currentTimeMillis();
            for(int n = 0; n < 400; n++){
                
                double angle = Math.random()*180.0;
                BufferedImage   image = producer.createImage(200, 200, Math.toRadians(angle), 0.1);
                INDArray        input = iter.getData(image);
                INDArray       output = network.output(input);
                
                int       maxBin = 0;
                double  maxValue = 0.0;
                
                for(int i = 0; i < 180; i++){
                    double value = output.getDouble(i);
                    if(value>maxValue){
                        maxValue = value;
                        maxBin   = i;
                    }                    
                }
                //System.out.println(maxValue + " bin = " + maxBin + "  angle = " + angle);
                System.out.println(String.format("%8.4f %8.5f %8.5f",(double) maxBin, angle, maxBin-angle));
            }
            long endTime = System.currentTimeMillis();
            System.out.println(String.format("elapsed time = %d", endTime-startTime));
            //System.out.println("output = " + output);
        } catch (IOException ex) {
            Logger.getLogger(RunEvaluation.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}
