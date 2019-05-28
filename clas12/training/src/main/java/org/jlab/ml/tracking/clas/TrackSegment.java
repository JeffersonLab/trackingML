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
public class TrackSegment implements Comparable<TrackSegment> {
    
    private Integer    trackID = -1;
    private Integer  clusterID = -1;
    private Integer     region = -1;
    
    private final double[][] segmentData = new double[6][112];
    
    public TrackSegment(int cid, int trkid){
        clusterID = cid;
        trackID   = trkid;
        reset();
    }
    
    public final void reset(){
        for(int l = 0; l < 6; l++)
            for(int w = 0; w < 112; w++) segmentData[l][w] = 0.0;
    }
    
    public int getClusterID(){
        return clusterID;
    }
    
    public int getTrackID(){ return trackID;}
    
    public void setRegion(int r){
        region = r;
    }
    
    public int getRegion(){
        return region;
    }
    
    public void setWire(int layer, int wire, double value){
        segmentData[layer][wire] = value;
    }
    
    public double getWire(int layer, int wire){
        return segmentData[layer][wire];
    }
    
    //public int getTrackID(){ return trackID;}
    public void setTrackID(int id){ trackID = id;}
    
    @Override
    public String toString(){
        StringBuilder str = new StringBuilder();
        str.append(String.format("CLUSTER %5d track = %3d region = %3d",clusterID,trackID,region));
        return str.toString();
    }

    @Override
    public int compareTo(TrackSegment o) {
       if(o.getTrackID()!=getTrackID()) return trackID.compareTo(o.trackID);
       return region.compareTo(o.region);
    }
}
