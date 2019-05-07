/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.jlab.ml.tracking.clas;

/**
 *
 * @author gavalian
 */
public class TrackSegment {
    
    private int trackID = 0;
    private final double[][] segmentData = new double[12][112];
    
    public TrackSegment(){
        
    }
    
    public void reset(){
        for(int l = 0; l < 12; l++)
            for(int w = 0; w < 112; w++) segmentData[l][w] = 0.0;
    }
    
    public void setWire(int layer, int wire, double value){
        segmentData[layer][wire] = value;
    }
    
    public double getWire(int layer, int wire){
        return segmentData[layer][wire];
    }
    
    public int getTrackID(){ return trackID;}
    public void setTrackID(int id){ trackID = id;}
}
