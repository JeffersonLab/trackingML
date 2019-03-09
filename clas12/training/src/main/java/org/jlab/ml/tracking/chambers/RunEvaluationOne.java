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
import org.jlab.groot.fitter.DataFitter;
import org.jlab.groot.math.F1D;
import org.jlab.groot.ui.TCanvas;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author gavalian
 */
public class RunEvaluationOne {
    public static void main(String[] args){
        // File file = new File("clas12_tracking_RELU.nnet");
       File file = new File("network_chambers_1k_epoch.nnet");
        int  nIterations = 500;
        
        
        long totalTime = 0L;
        try {
            
            MultiLayerNetwork network = ModelSerializer.restoreMultiLayerNetwork(file);
            
            System.out.println(network.summary());
            
            String score = "SCORE = " + String.format("%.8f", network.score());
            System.out.println("\n>>>>>>>>>>>>>>> END OF SUMMARY <<<<<<<<<<<<<");
            System.out.println(score);
            TCanvas c1 = new TCanvas("c1",1100,500);
            c1.getCanvas().initTimer(4000);
            GraphErrors graph = new GraphErrors();
            H1F         histo = new H1F("h1",300,-2.0,2.0);
            
            c1.divide(2,1);
            c1.cd(0);
            c1.getPad().setAxisFontSize(24);
            c1.getPad().setAxisLabelFontSize(24);
            c1.draw(graph);
            c1.cd(1);
            c1.getPad().setTitle(score);
            c1.getPad().setAxisFontSize(24);
            c1.getPad().setAxisLabelFontSize(24);
            c1.getPad().setStatBoxFontSize(18);
            c1.getPad().setStatBoxFont("Times New Roman");
            c1.draw(histo);
            c1.getPad().setOptStat("11111111111111");
            histo.setLineColor(4).setFillColor(44);
            graph.setMarkerSize(3);
            graph.setMarkerColor(2);
            
            histo.setTitle(score);
            histo.setTitleX("#theta-#theta'");
            graph.setTitleX("#Theta");
            graph.setTitleY("#Theta'");
            
            long total_time   = 0L;
            long total_nano   = 0L;
            //long start_time = System.currentTimeMillis();
            long start_time = System.nanoTime();
            for(int i = 0; i < nIterations; i++){
                DataLoader loader = new DataLoader();
                loader.generate(1);
                INDArray input = loader.getInputArray();
                long st_nano = System.nanoTime();
                INDArray output = network.output(input);
                long et_nano = System.nanoTime();
                total_nano += et_nano - st_nano;
                /*for(int t = 0; t < 5000; t++){
                    output = network.output(input);
                }*/
                
                
                //System.out.println(String.format(" TIME = %9.3f ", (end_time-start_time)/5000.0) );
                double angle = loader.getDetectorHits().get(0).getAngle();
                double angleOut = output.getDouble(0, 0)*20.0-10.0;
                //System.out.println(String.format("%8.5f %8.5f", angle ,angleOut));
                graph.addPoint(angle, angleOut, 0.0, 0.0);
                 histo.fill((angle-angleOut));
            }
            //long end_time = System.currentTimeMillis();
            long end_time = System.nanoTime();
            total_time += end_time - start_time;
            F1D funcg = new F1D("funcg","[amp]*gaus(x,[mean],[sigma])",-1.0,1.0);
            funcg.setParameter(0, 20);
            funcg.setParameter(1, 0.0);
            funcg.setParameter(2, 0.2);
            funcg.setLineStyle(4);
            funcg.setLineWidth(3);
            DataFitter.fit(funcg, histo, "");
            c1.cd(1);
            c1.draw(funcg,"same");
            System.out.println("SCORE = " + network.score());
            System.out.println("TOTAL TIME = " + total_time + " NANO = " + total_nano);
            System.out.println(String.format("TIME PER EVAL = %9.3f", total_nano/((double)nIterations)/1000000.0));
            } catch (IOException ex) {
            Logger.getLogger(org.jlab.ml.tracking.nn.RunEvaluation.class.getName()).log(Level.SEVERE, null, ex);
        
            }
        
        
    }
}
