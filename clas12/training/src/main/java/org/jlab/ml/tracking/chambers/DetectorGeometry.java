/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.jlab.ml.tracking.chambers;

import org.jlab.jnp.geom.prim.Line3D;
import org.jlab.jnp.geom.prim.Path3D;

/**
 *
 * @author gavalian
 */
public class DetectorGeometry {
    
    private double[] chamberPositions = new double[]{20.0,40.0,60.0,80.0,100.0,120.0};
    private double   layerWidth       = 18.0;
    private double   wireLength       = 30.0;
    private double[][] chamberBuffer  = new double[36][112];
    private double     trackAngleDegrees = 0.0;
    
    
    public DetectorGeometry(){
        
    }
    
    
    public void setAngle(double angle){
        this.trackAngleDegrees = angle;
    }
    
    public double getAngle(){
        return trackAngleDegrees;
    }
    
    public void processStraight(){
      processStraight(trackAngleDegrees);  
    }
    
    public void processStraight(double angle){
        setAngle(angle);
        reset();
        double angle_rad = Math.toRadians(angle);
        double r         = 800.0;
        Line3D     track = new Line3D(0.0,0.0,0.0,r*Math.sin(angle_rad),0.0,r*Math.cos(angle_rad));
        Path3D     trackPath = new Path3D();
        trackPath.addPoint(track.origin());
        trackPath.addPoint(track.end());        
        getHits(trackPath);
    }
    
    public final double[][] getBuffer(){ return chamberBuffer;}
            
    public void reset(){
        for(int l = 0; l < 36; l++)
            for(int w = 0; w < 112; w++) chamberBuffer[l][w]=0.0;
    }
    
    public double getDistance(Path3D path, int layer, int wire){
        double  distance = 10000.0;
        int    nLines   = path.getNumLines();
        for(int l = 0; l < nLines; l++){
            Line3D str8L = path.getLine(l);
            Line3D wireL = getWire(layer,wire);
            Line3D    d = str8L.distanceSegments(wireL);
            if(d.length()<distance){
                distance = d.length();
            }
        }
        return distance;
    }
    
    public int getClosestWire(Path3D path, int layer){
        double  distance = 10000.0;
        int    wireIndex = -1;       
        for(int i = 0; i < 112; i++){            
            double    dst = getDistance(path,layer,i);
            if(dst<distance){
                distance = dst;
                wireIndex = i; 
            }
        }
        return wireIndex;
    }
    
    public void getHits(Path3D path){
        
        for(int layer = 0; layer < 36; layer++){
            int   index = getClosestWire(path,layer);
            double dist = getDistance(path,layer,index);
            //System.out.println(String.format("%5d %5d distance = %8.5f", 
            //        layer,index,dist));
            if(dist<0.51) chamberBuffer[layer][index] = 1.0;
        }
    }
    
    public Line3D getWire(int layer, int wire){
        int index = layer/6;
        //System.out.println(" layer = " + layer + " block = " + index);
        double z_position = chamberPositions[index] + ((double) wire);
        double x_position = 56.5 - wire;
        return new Line3D(x_position, -wireLength/2.0, z_position,
                          x_position,  wireLength/2.0, z_position);
    }
    
    public static void main(String[] arg){
        DetectorGeometry  geom = new DetectorGeometry();
        geom.processStraight(25.0);
    }
}
