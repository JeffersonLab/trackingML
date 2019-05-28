/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.jlab.ml.tracking.clas;

import java.util.ArrayList;
import java.util.Collections;
import  org.jlab.jnp.hipo4.data.*;
import  org.jlab.jnp.hipo4.io.*;
import  java.util.Map;
import java.util.HashMap;
import java.util.List;

/**
 *
 * @author gavalian
 */
public class ProcessData {
    
    
    private int[][] sortingOrder = new int[][]{
        { 0, 7, 8, 9, 10, 11},
        { 0, 1, 8, 9, 10, 11},
        { 0, 1, 2, 9, 10, 11},
        { 0, 1, 2, 3, 10, 11},
        { 0, 1, 2, 3,  4, 11},
        { 6, 1, 2, 3,  4,  5},
        { 6, 7, 2, 3,  4,  5},
        { 6, 7, 8, 3,  4,  5},
        { 6, 7, 8, 9,  4,  5},
        { 6, 7, 8, 9, 10,  5}        
    };
    
    Map<Integer,TrackSegment> clusters = new HashMap<Integer,TrackSegment>();
    
    
    public void processEvent(Bank hits){
        int rows = hits.getRows();
        //System.out.println("rows = " + rows);
        for(int i = 0; i < rows; i++){
            int    wire = hits.getInt("wire", i);
            int  region = hits.getInt("superlayer", i);
            int   layer = hits.getInt("layer", i);
            int cluster = hits.getInt("clusterID", i);
            int  sector = hits.getInt("sector", i);
            int   track = hits.getInt("trkID", i);
            if(sector==1&&track>0){
                if(clusters.containsKey(cluster)==false){
                    TrackSegment segment = new TrackSegment(cluster,track);
                    clusters.put(cluster,segment);
                }
            
                TrackSegment sg = clusters.get(cluster);
                sg.setRegion(region);
                //System.out.println("setting L = " + layer + " wire = " + wire);
                sg.setWire(layer-1, wire-1, 1.0);
            }
        }
        //System.out.println("size = " + clusters.size());
    }
    public int getSize(){ return clusters.size();}
    
    public void clear(){ clusters.clear();}
    public void show(){
        System.out.println(" EVENT # printout");
        for(Map.Entry<Integer,TrackSegment> entry : clusters.entrySet()){
            System.out.println(entry.getValue().toString());
        }
    }
    
    public void saveImage(String directory, int eventNumber){
        String filename = directory + "/dcimage_p_" + eventNumber + ".png";
        ImageProducer  producer = new ImageProducer();
        List<TrackSegment> list = getSegmentsList();
        producer.produceImage(list);
        producer.saveImage(filename);
    }
    
    public List<TrackSegment>  getSegmentsList(){
        List<TrackSegment> list = new ArrayList<TrackSegment>();
        for(Map.Entry<Integer,TrackSegment> entry : clusters.entrySet()){
            list.add(entry.getValue());
        }
        return list;
    }
    
    public void combinatorics(String dir , int eventNumber){
        List<TrackSegment> segments = getSegmentsList();
        System.out.println("------------------ before sort");
        for(int i = 0; i < segments.size(); i++){
            System.out.println(i + " : " + segments.get(i));
        }
        Collections.sort(segments);
        System.out.println("------------------ after sort");
        for(int i = 0; i < segments.size(); i++){
            System.out.println(i + " : " + segments.get(i));
        }
        System.out.println(" LENGTH = " + this.sortingOrder[0].length);
        for(int i = 0; i < 10; i++){
            System.out.println(" COMBINATORICS " + i);
            List<TrackSegment> list = getListWithIndex(segments,this.sortingOrder,i);
            for(int k = 0; k < list.size(); k++){
                System.out.println(k + " : " + list.get(k));
            }
            String filename = dir + "/dcimage_n_" + eventNumber + "_" + i + ".png";
            ImageProducer  producer = new ImageProducer();
            producer.produceImage(list);
            producer.saveImage(filename);
        }
    }
    
    public List<TrackSegment>  getListWithIndex(List<TrackSegment> list, int[][] index, int order){
        List<TrackSegment> array = new ArrayList<TrackSegment>();
        for(int i = 0; i < 6; i++){
            array.add(list.get(index[order][i]));
        }
        return array;
    }
    
    public static void main(String[] args){
        String filename = "/Users/gavalian/Work/Software/project-7a.0.0/clas_004013_ML.hipo";
        
        HipoReader reader = new HipoReader();
        ProcessData data  = new ProcessData();
        
        reader.open(filename);
        Event hipoEvent = new Event();
        Bank  dcHits    = new Bank(reader.getSchemaFactory().getSchema("TimeBasedTrkg::TBHits"));
        int counter = 0;
        while(reader.hasNext()==true){
            //System.out.println(" event = " + counter);
            reader.nextEvent(hipoEvent);
            hipoEvent.read(dcHits);
            data.processEvent(dcHits);

            counter++;
            if(data.getSize()==6){
                data.show();
                data.saveImage("dctrack", counter);
                //counter++;
            }
            
            if(data.getSize()==12){
                data.show();
                data.combinatorics("dctrack",counter);
                //data.saveImage("dctrack", counter);
                //counter++;
            }
            data.clear();
        }
        System.out.println("processed event # = " + counter);
    }
}
