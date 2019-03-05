/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.jlab.ml.tracking.clas12;

/**
 *
 * @author gavalian
 */
public class TrackData {
    private float[][] matrix = null;
    private float[]   trackParams = null;
    
    private double    trackP = 0.0;
    private double    trackTheta = 0.0;
    private double    trackPhi = 0.0;
    

    public TrackData(){
        matrix = new float[36][112];
        trackParams = new float[3];
    }
    
    public void reset(){
        for(int layer = 0; layer < 36; layer++)
            for(int wire = 0; wire < 112; wire++){
                matrix[layer][wire] = 0.0f;
            }
    }

    public void setWire(int layer, int wire){
        matrix[layer][wire] = 1.0f;
    }
    
    public float[][] getWires(){ return matrix;}
    
    public boolean setPxPyPz(double px, double py, double pz){        
        trackP     = Math.sqrt(px*px + py*py + pz*pz);
        if(trackP<1.0||trackP>11.0) return false;
        trackTheta = Math.toDegrees(Math.acos(pz/trackP));
        if(trackTheta<10.0||trackTheta>25) return false; 
        trackPhi   = Math.toDegrees(Math.atan2(py, px));
        if(trackPhi<-30.0||trackPhi>30.0) return false;
       
        trackParams[0] = (float) ((trackP-1.0));
        trackParams[1] = (float) (pz/trackP);
        trackParams[2] = (float) ((trackPhi+30.0)/60.0);
        
        trackPhi += 30.0;
        return true;
    }
    
    public double getP(){ return trackP;}
    public double getTheta() { return trackTheta;}
    public double getPhi() { return trackPhi;}
    
    public float[] parameters(){ return trackParams;}

}
