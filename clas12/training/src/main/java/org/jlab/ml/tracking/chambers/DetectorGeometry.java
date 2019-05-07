/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.jlab.ml.tracking.chambers;

import java.util.ArrayList;
import java.util.List;
import org.jlab.groot.data.H2F;
import org.jlab.groot.ui.TCanvas;
import org.jlab.jnp.geom.prim.Line3D;
import org.jlab.jnp.geom.prim.Path3D;

/**
 *
 * @author gavalian
 */
public class DetectorGeometry {
    
    private double[] chamberPositions = new double[]{30.0,80.0,130.0,180.0,230.0,280.0};
    private double   layerWidth       = 18.0;
    private double   wireLength       = 30.0;
    private double[][] chamberBuffer  = new double[36][112];
    private double     trackAngleDegrees = 0.0;
    private double     vertexMin         = 0.0;
    private double     vertexMax         = 0.0;
    private double     vertexZcoord      = 0.0;
    
    public DetectorGeometry(){
        
    }
    
    
    public void setVertex(double min, double max){
        this.vertexMin = min;
        this.vertexMax = max;
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
        vertexZcoord   = vertexMin + Math.random()*(vertexMax-vertexMin);
        Line3D     track = new Line3D(0.0,0.0,vertexZcoord,
                r*Math.sin(angle_rad),0.0,
                vertexZcoord+r*Math.cos(angle_rad));
        Path3D     trackPath = new Path3D();
        trackPath.addPoint(track.origin());
        trackPath.addPoint(track.end());        
        getHits(trackPath);
    }
    
    public double getVertex(){ return this.vertexZcoord;}
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
            if(dist<0.99) chamberBuffer[layer][index] = 1.0;
            //if(dist<0.99) chamberBuffer[layer][index] = 1.0-dist;
        }
    }
    
    public H2F getH2F(){
        H2F h = new H2F("h2",36,-0.5,35.5,112,-0.5,111.5);
        for(int layer = 0; layer < 36; layer++){
            for(int wire = 0; wire < 112; wire++){
                h.setBinContent(layer, wire, chamberBuffer[layer][wire]);
            }
        }
        return h;
    }
    
    public Line3D getWire(int layer, int wire){
        int index = layer/6;
        int localLayer = layer - 6*index;
        //System.out.println(" layer = " + layer + " block = " + index);
        double z_position = chamberPositions[index] + ((double) 3.0*localLayer);
        double x_position = 56.5 - wire;
        
        //System.out.println(" layer = " + layer + " block = " + index +
        //        String.format(" z = %8.5f ",z_position));
        return new Line3D(x_position, -wireLength/2.0, z_position,
                          x_position,  wireLength/2.0, z_position);
    }
    
    public static void main(String[] arg){
        
        /*List<Line3D> lines = new ArrayList<Line3D>();
        DetectorGeometry  geom = new DetectorGeometry();
        
        for(int i = 0; i < 10; i++){
            double z = 50.0 + 2.0*i;
            Line3D line = new Line3D(25.0,-30,z,25.0,30,z);
            lines.add(line);
        }
        double r = 200.0;
        double a = Math.toRadians(23);
        Line3D path = new Line3D(0.0,0.0,0.0, r*Math.sin(a),0.0,r*Math.cos(a));
        path.show();
        
        for(int i = 0; i < lines.size(); i++){
            Line3D dist = path.distanceSegments(lines.get(i));
            System.out.println(i + " " + dist.length());            
        }
        System.out.println("\n\n============>>>>>>");
        for(int layer = 0; layer < 12; layer++){
            Line3D wire = geom.getWire(layer, 20);
            //wire.show();
            Line3D dist = path.distanceSegments(wire);
            System.out.println(layer + " " + dist.length()); 
        }*/
        
        DetectorGeometry  geom = new DetectorGeometry();
        geom.setAngle(-22);
        geom.processStraight();
        geom.setVertex(0, 0);
        TCanvas c1 = new TCanvas("c1",500,500);
        
        c1.divide(4, 4);
        for(int i = 0; i < 16; i++){
            Integer angle = 0+i;//-10+i;
            geom.setAngle(angle);
            geom.processStraight();
            H2F histo = geom.getH2F();
            histo.setTitle("ANGLE = " + angle.toString() 
                    + " VERTEX = " + String.format("%.3f", geom.getVertex()));
            c1.cd(i);
            c1.draw(histo);
        }
    }
}
