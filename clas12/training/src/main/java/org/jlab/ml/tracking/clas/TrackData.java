/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.jlab.ml.tracking.clas;

import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author gavalian
 */
public class TrackData {
    
    private final List<TrackSegment>  segmentsList = new ArrayList<TrackSegment>();
    
    public TrackData(){
        
    }
    
    public void addSegment(TrackSegment seg){
        segmentsList.add(seg);
    }
    
    public TrackSegment getSegment(int index){
        return this.segmentsList.get(index);
    }
    
}
