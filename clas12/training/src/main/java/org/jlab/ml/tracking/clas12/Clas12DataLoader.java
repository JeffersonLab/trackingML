/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.jlab.ml.tracking.clas12;

import java.util.ArrayList;
import java.util.List;
import org.jlab.groot.math.Axis;
import org.jlab.groot.math.MultiIndex;
import org.jlab.jnp.hipo4.data.Bank;
import org.jlab.jnp.hipo4.data.Event;
import org.jlab.jnp.hipo4.io.HipoReader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author gavalian
 */
public class Clas12DataLoader {
    
    private List<TrackData>  trackData = new ArrayList<TrackData>();
    private Axis             axisP = null;
    private Axis             axisTheta = null;
    private Axis             axisPhi = null;
    private MultiIndex       index   = null;
    
    public Clas12DataLoader(){
       axisP = new Axis(1000,1.0,11.0);
       axisTheta = new Axis(15,10.0,25.0);
       axisPhi   = new Axis(30,0.0,60.0);
       index = new MultiIndex(100,15,30);
       System.out.println("offset = " + index.getArrayIndex(1,1,1));
    }
    
    
    public List<TrackData> tracks(){ return trackData;}
    
    public INDArray getInputArray(int size){
        int[]   inputs   = new int[]{size,1,36,112};
        int bufferLength = size*36*112;
        float[] buffer   = new float[bufferLength];
        INDArray ind = Nd4j.create(buffer, inputs);
        for(int i = 0; i < size; i++){
            TrackData data = this.trackData.get(i);
            float[][] matrix = data.getWires();
            for(int l = 0; l < 36; l++){
                for(int w = 0; w < 112; w++){
                    ind.putScalar(i, 0, l, w, matrix[l][w]);
                }
            }
        }
        return ind;
    }
    
    public INDArray getInputArray(List<TrackData> data){
        int size = data.size();
        int[]   inputs   = new int[]{size,1,36,112};
        int bufferLength = size*36*112;
        float[] buffer   = new float[bufferLength];
        INDArray ind = Nd4j.create(buffer, inputs);
        for(int i = 0; i < size; i++){
            TrackData trk = data.get(i);
            float[][] matrix = trk.getWires();
            for(int l = 0; l < 36; l++){
                for(int w = 0; w < 112; w++){
                    ind.putScalar(i, 0, l, w, matrix[l][w]);
                }
            }
        }
        return ind;
    }
    
    public INDArray getOutputArray(int size){
        int outSize = index.getArraySize();
        
        int[] inputs = new int[]{size,1000};
        
        int   bufferLength = size*1000;
        float[] buffer = new float[bufferLength];
        INDArray ind = Nd4j.create(buffer, inputs);
        
        for(int i = 0; i < size; i++){
            TrackData data = this.trackData.get(i);
            int offset  = axisP.getBin(data.getP());
            
            for(int k = 0; k < 1000; k++) ind.putScalar(i, k, 0.0);
            //System.out.println(String.format("%5d %5d %5d : %5d", idx_p,idx_th,idx_phi,offset));
            ind.putScalar(i, offset, 1.0);
            //ind.putScalar(i, 1, data.parameters()[1]);
            //ind.putScalar(i, 2, data.parameters()[2]);
            //System.out.println(String.format("%8.5f %8.5f %8.5f", ind.getDouble(i,0),
            //        ind.getDouble(i,1),ind.getDouble(i,2))
                    //data.parameters()[0],data.parameters()[1],data.parameters()[2])
            //);
            //ind.putScalar(i, offset, 1.0);
            //for(int p = 0; p < 3; p++){
            //    ind.putScalar(i, p , data.parameters()[p]);
            //}
        }
        return ind;
    }
    
    public INDArray getOutputArray(List<TrackData> data){
        int size = data.size();
        
        int outSize = index.getArraySize();
        
        int[] inputs = new int[]{size,3};
        
        int   bufferLength = size*3;
        float[] buffer = new float[bufferLength];
        INDArray ind = Nd4j.create(buffer, inputs);
        
        for(int i = 0; i < size; i++){
            TrackData trk = data.get(i);
            int idx_p  = axisP.getBin(trk.getP());
            int idx_th = axisTheta.getBin(trk.getTheta());
            int idx_phi = axisPhi.getBin(trk.getPhi());
            
            int offset = index.getArrayIndex(idx_p,idx_th,idx_phi);
            ind.putScalar(i, 0, trk.parameters()[0]);
            ind.putScalar(i, 1, trk.parameters()[1]);
            ind.putScalar(i, 2, trk.parameters()[2]);
            //for(int k = 0; k < outSize; k++) ind.putScalar(i, k, 0.0);
            //System.out.println(String.format("%5d %5d %5d : %5d", idx_p,idx_th,idx_phi,offset));
            //ind.putScalar(i, offset, 1.0);
            //for(int p = 0; p < 3; p++){
            //    ind.putScalar(i, p , data.parameters()[p]);
            //}
        }
        return ind;
    }
    
    public void readFile(String filename){
        HipoReader reader = new HipoReader();
        reader.open(filename);
        
        int nevents = reader.getEventCount();
        int counter = 0;
        Event event = new Event();
        
        TrackData   data = new TrackData();
        
        Bank  TBHits = new Bank(reader.getSchemaFactory().getSchema("TimeBasedTrkg::TBHits"));
        Bank  Tracks = new Bank(reader.getSchemaFactory().getSchema("TimeBasedTrkg::TBTracks"));
        while(reader.hasNext()==true){
            reader.nextEvent(event);            
            
            event.read(Tracks);
            event.read(TBHits);
            
            int ntracks = Tracks.getRows();
            int nsector = 0;
            int trackid = 0;
            int trkstatus  = 0;
            
            double px   = 0.0;
            double py   = 0.0;
            double pz   = 0.0;
            for(int t = 0; t < ntracks; t++){
                int sector = Tracks.getInt("sector", t);
                int charge = Tracks.getInt("q", t);
                int status = Tracks.getInt("status", t);
                //System.out.println(" : " + sector + " " + charge);
                if(sector==1&&charge<0){
                    nsector++;
                    trackid = t+1;
                    if ((status==110)||status==100) {
                        trkstatus = 1;
                    }
 
                    px = Tracks.getFloat("p0_x", t);
                    py = Tracks.getFloat("p0_y", t);
                    pz = Tracks.getFloat("p0_z", t);
                }
            }
            
            if(nsector==1&&trkstatus>0){
                boolean status = data.setPxPyPz(px, py, pz);
                if(status==true){
                    counter++;
                
                    int nhits = TBHits.getRows();
                    TrackData  trk = new TrackData();
                    trk.setPxPyPz(px, py, pz);
                    trk.reset();
                    for(int h = 0; h < nhits; h++){
                        int sector = TBHits.getInt("sector", h);
                        int trkid  = TBHits.getInt("trkID", h);
                        if(sector==1&&trkid==trackid){
                            int layer = TBHits.getInt("layer", h);
                            int superlayer = TBHits.getInt("superlayer", h);
                            int wire = TBHits.getInt("wire", h);
                            int l = (superlayer-1)*6 + (layer-1);
                            trk.setWire(l, wire-1);
                        }
                    }
                    trackData.add(trk);
                }
            }
            
        }
        System.out.println("# of tracks = " + counter + "  data = " + this.trackData.size() );
    }
    
    
    public int[] getIndex(int offset){
        for(int i1 = 0; i1 < 100; i1++){
            for(int i2 = 0; i2 < 15; i2++){
                for(int i3 = 0; i3 < 30 ; i3++){
                    int position = index.getArrayIndex(i1,i2,i3);
                    if(position==offset){
                        return new int[]{i1,i2,i3};
                    }
                }
            }
        }
        return new int[]{-1,-1,-1};
    }
    
    public double evaluateTheta(INDArray output){
    
        int counter = 0;
        double pw = 0.0;
        double w  = 0.0;
        
        for(int i = 0; i < index.getArraySize(); i++){
            double value = output.getDouble(0,i);
            int[] ids = getIndex(i);
            if(value>0.0001){
                double p = axisTheta.getBinCenter(ids[1]);
                pw += p*value;
                w  += value;
            }
        }
        //System.out.println(" bins = " + counter);
        return pw/w;
    }
    
    public double evaluatePhi(INDArray output){
    
        int counter = 0;
        double pw = 0.0;
        double w  = 0.0;
        
        for(int i = 0; i < index.getArraySize(); i++){
            double value = output.getDouble(0,i);
            int[] ids = getIndex(i);
            if(value>0.0001){
                double p = axisPhi.getBinCenter(ids[2]);
                pw += p*value;
                w  += value;
            }
        }
        //System.out.println(" bins = " + counter);
        return pw/w;
    }
    
    public double evaluateP(INDArray output){
        int counter = 0;
        double pw = 0.0;
        double w  = 0.0;
        
        for(int i = 0; i < index.getArraySize(); i++){
            double value = output.getDouble(0,i);
            int[] ids = getIndex(i);
            if(value>0.0001){
                double p = axisP.getBinCenter(ids[0]);
                pw += p*value;
                w  += value;
            }
        }
        //System.out.println(" bins = " + counter);
        return pw/w;
    }
    public static void main(String[] args){
        String filename = "/Users/gavalian/Work/Software/project-6a.0.0/clas_004013_ML.hipo";
        Clas12DataLoader loader = new Clas12DataLoader();
        loader.readFile(filename);
        loader.getOutputArray(40);
    }
}
