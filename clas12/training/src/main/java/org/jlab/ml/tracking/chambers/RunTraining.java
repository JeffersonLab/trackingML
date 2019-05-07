/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.jlab.ml.tracking.chambers;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.optimize.listeners.TimeIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.jlab.jnp.readers.TextFileWriter;
import org.jlab.ml.tracking.clas12.Clas12DataLoader;
import org.jlab.ml.tracking.clas12.Clas12TrackingModel;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author gavalian
 */
public class RunTraining {
    
    private int nSamples = 1000;
    private int nEpochs  = 100;
    private int saveFrequency = 50;
    
    private String directory = "cnnet";
    private RunEvaluationOne evaluation = new RunEvaluationOne();
    private MultiLayerNetwork network = null;
    private Map<Integer,Double> epochScore = new LinkedHashMap<Integer,Double>();
    
    
    
    public RunTraining(){ 
    }
    
    
    public void saveProgress(int epoch){
        String name = directory + String.format("/progress_%d.score", epoch);
        TextFileWriter  writer = new TextFileWriter();
        writer.open(name);
        double[] output = new double[2];
        
        for(Map.Entry<Integer,Double> entry : epochScore.entrySet()){
            output[0] = entry.getKey();
            output[1] = entry.getValue();
            String strOut = String.format("%e %e", output[0],output[1]);
            writer.writeString(strOut);
        }
        writer.close();
    }
    public void setDirectory(String outputDir){ 
        directory = outputDir; 
        evaluation.setDirectory(outputDir);
    }
    
    public void setNSamples(int samples) { nSamples = samples;}
    public void setNEpochs(int epochs) { nEpochs = epochs; }
    public void setSaveFrequency(int freq){ saveFrequency = freq;}
    
    public void run(){
        
        DataLoader loader = new DataLoader();
        loader.generateH(nSamples);
        
        INDArray  aInputs = loader.getInputArray(   );
        INDArray aOutputs = loader.getOutputArrayOne(  );
        
        
        
        ChamberModel model = new ChamberModel();
        
        MultiLayerConfiguration configuration = model.getConfiguration();
                
        network = new MultiLayerNetwork(configuration);        
        network.init();
        System.out.println(network.summary());
        for(int epoch = 1; epoch <= nEpochs; epoch++){
            System.out.println(String.format("> starting epoch %8d/%8d", epoch,nEpochs));
            long start_time = System.currentTimeMillis();
            network.fit(aInputs, aOutputs);
            long end_time   = System.currentTimeMillis();
            long elapsed_time = end_time - start_time;
            double score = network.score();
            System.out.println(String.format(">>> result epoch %8d/%8d score = %.14f time = %8d ms", 
                    epoch,nEpochs, score, elapsed_time));
            this.epochScore.put(epoch, score);
            if(epoch%saveFrequency==0){
                System.out.println(">>>> saving network and running evaluation....");
                saveNetwork(epoch);
                saveProgress(epoch);
                evaluation.run(network, epoch, 2000);
            }
        }
    }
    
    
    public void saveNetwork(int epoch){
        String filename = directory + String.format("/chamber_cnnet_%d.nnet", epoch);
         try {            
            ModelSerializer.writeModel(network, filename, true);                                   
        } catch (IOException ex) {
            Logger.getLogger(org.jlab.ml.tracking.nn.RunTraining.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    public static void main(String[] args){
        
        System.out.println("running ML tracking learning algorithm...");
                
        int  nIterations = Integer.parseInt(args[0]);
        int nDataSamples = Integer.parseInt(args[1]);
        int nSaveFrequency = Integer.parseInt(args[2]);
        String directory   = args[3];
        
        RunTraining training = new RunTraining();
        
        training.setNEpochs(nIterations);
        training.setNSamples(nDataSamples);
        training.setSaveFrequency(nSaveFrequency);
        training.setDirectory(directory);
        
        
        training.run();
        
        /*
        System.out.println(" SET ITERATIONS   = " + nIterations);
        System.out.println(" SET DATA SAMPLES = " + nDataSamples);
        
        ChamberModel model = new ChamberModel();
        
        MultiLayerConfiguration configuration = model.getConfiguration();
                
        MultiLayerNetwork network = new MultiLayerNetwork(configuration);        
        network.init();
        
        
        System.out.println(network.summary());
        network.addListeners(new ScoreIterationListener(1));        
        network.addListeners(new TimeIterationListener(1));

        
        for(int i = 0; i < names.size(); i++){
            System.out.println(String.format("%3d : %s", i, names.get(i)));
        }
        
        for(int i = 0; i < nIterations; i++){
            

            DataLoader loader = new DataLoader();
            loader.generate(nDataSamples);
            
            INDArray  aInputs = loader.getInputArray(   );
            INDArray aOutputs = loader.getOutputArrayOne(  );
            
            System.out.println("running epoch # " + i);
            System.out.println(">> generating new sample with # " + nDataSamples);
            long start_time = System.currentTimeMillis();
            network.fit(aInputs, aOutputs);
            long end_time   = System.currentTimeMillis();
            long elapsed_time = end_time - start_time;
            double score = network.score();
            System.out.println("epoch # " + i  + "/" + nIterations 
                    + " is complete with score = " + String.format("%.12f", score) + " time = " 
                    + elapsed_time + " ms");
        }
                              
        try {
            
            ModelSerializer.writeModel(network, "network_chambers.nnet", true);
                        
           
        } catch (IOException ex) {
            Logger.getLogger(org.jlab.ml.tracking.nn.RunTraining.class.getName()).log(Level.SEVERE, null, ex);
        }*/
    }
}
