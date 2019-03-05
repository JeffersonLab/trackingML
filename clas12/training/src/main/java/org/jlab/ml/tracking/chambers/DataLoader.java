/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.jlab.ml.tracking.chambers;

import java.util.ArrayList;
import java.util.List;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.jlab.groot.math.Axis;

/**
 *
 * @author gavalian
 */
public class DataLoader {
    
    List<DetectorGeometry>  detectorHits = new ArrayList<DetectorGeometry>();
    
    public DataLoader(){ }
    
    
    public List<DetectorGeometry> getDetectorHits(){ return detectorHits;}
    
    public void addData(DetectorGeometry dg){ detectorHits.add(dg);}
    
    public INDArray getInputArray(){
        int size = detectorHits.size();
        int[]   inputs   = new int[]{size,1,36,112};
        int bufferLength = size*36*112;
        float[] buffer   = new float[bufferLength];
        INDArray ind = Nd4j.create(buffer, inputs);
        for(int i = 0; i < size; i++){
            DetectorGeometry geom = detectorHits.get(i);
            for(int layer = 0; layer < 36; layer++){
                for(int wire = 0; wire < 112; wire++){
                    ind.putScalar(i, 0, layer, wire, geom.getBuffer()[layer][wire]);
                }
            }
        }
        return ind;
    }
    
    public INDArray getOutputArray(){
        int outSize = detectorHits.size();
        int bins    = 50;
        int[] inputs = new int[]{outSize,bins};
        Axis  theta  = new Axis(bins,-25.0,25.0);
        int   bufferLength = outSize*bins;
        float[] buffer = new float[bufferLength];
        INDArray ind = Nd4j.create(buffer, inputs);
        for(int i = 0; i < outSize; i++){
            DetectorGeometry geom = detectorHits.get(i);
            int index = theta.getBin(geom.getAngle());
            //System.out.println(" bin = " + index);
            for(int b = 0; b < bins; b++){
                ind.putScalar(i, b, 0.0);
                if(b == index) ind.putScalar(i, b, 1.0);
            }
        }
        return ind;
    }
    
    public INDArray getOutputArrayOne(){
        int outSize = detectorHits.size();
        int bins    = 1;
        int[] inputs = new int[]{outSize,bins};
        Axis  theta  = new Axis(bins,-25.0,25.0);
        int   bufferLength = outSize*bins;
        float[] buffer = new float[bufferLength];
        INDArray ind = Nd4j.create(buffer, inputs);
        for(int i = 0; i < outSize; i++){
            DetectorGeometry geom = detectorHits.get(i);
            double value = (10 + geom.getAngle())/20.0;
            ind.putScalar(i, 0, value);
        }
        return ind;
    }
    public void generate(int samples){
        for(int i = 0; i < samples; i++){
            double angle = Math.random()*20.0-10.0;
            DetectorGeometry geom = new DetectorGeometry();
            geom.setAngle(angle);
            geom.processStraight(angle);
            detectorHits.add(geom);
        }
    }
}
