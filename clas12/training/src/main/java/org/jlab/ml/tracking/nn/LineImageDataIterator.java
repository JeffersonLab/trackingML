/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.jlab.ml.tracking.nn;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.imageio.ImageIO;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author gavalian
 */
public class LineImageDataIterator {
    
    private List<String>       inputFiles = new ArrayList<String>();
    private int           numberOfClasses = 180;
    
    public LineImageDataIterator(){
        
    }
    
    public void addFiles(List<String> files){
        this.inputFiles.clear();
        for(String file : files){
            inputFiles.add(file);
        }
        System.out.println("[data iterator] ----> added " + files.size() + " files.");
    }
    
    
    public void readDirectory(String directory){
        Filewalker walker = new Filewalker();
        walker.setFilterExtension(".png");
        walker.walk(directory);

        addFiles(walker.getList());
    }
    
    public void addImageFile(String file){
        
    }
    
    protected int getImageClass(String image){
        String[] tokens = image.split("/");
        //System.out.println(" tokens = " + tokens.length);
        if(tokens.length>=2){
            int index = tokens.length-2;
            //System.out.println(" id = " + tokens[index]);
            return Integer.parseInt(tokens[index]);
        }
        return 0;
    }
    
    public void imageToBuffer(String image, float[] buffer){
        File imageFile = new File(image);
        try {
            
            BufferedImage bi = ImageIO.read(imageFile);
            int xsize = bi.getWidth();
            int ysize = bi.getHeight();
            
            //System.out.println(String.format("[image] open : width = %6d, height = %6d",xsize,ysize));
            int counter = 0;
            for(int x = 0; x < xsize; x++){
                for(int y = 0; y < ysize; y++){
                    int rgb = bi.getRGB(x, y);
                    if((rgb&0x00ffffff)==0){
                        buffer[counter] = 0.0f;
                    } else {
                        buffer[counter] = 1.0f;
                    }
                    /*if(buffer[counter]>0.5)
                        System.out.println(String.format("x: %5d, y: %5d -->> %8.4f", x,y,buffer[counter]));*/
                    counter++;
                }
            }
        } catch (IOException ex) {
            Logger.getLogger(LineImageDataIterator.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    public INDArray getData(BufferedImage bi){
        int xsize = bi.getWidth();
        int ysize = bi.getHeight();
        int[] inputs = new int[]{1,1,xsize,ysize};
        float[] buffer   = new float[xsize*ysize];
        INDArray ind = Nd4j.create(buffer, inputs);
        int counter = 0;
        float value = 1.0f;
        for(int x = 0; x < xsize; x++){
            for(int y = 0; y < ysize; y++){
                int rgb = bi.getRGB(x, y);
                if((rgb&0x00ffffff)==0){
                    value = 0.0f;
                } else {
                    value = 1.0f;
                }
                ind.putScalar(0,0, x, y, value);
                /*if(buffer[counter]>0.5)
                System.out.println(String.format("x: %5d, y: %5d -->> %8.4f", x,y,buffer[counter]));*/
                counter++;
            }
        }
        return ind;
    }
    
    public INDArray getInputArray(){
        
        int count = inputFiles.size();
        int bufferLength = count*200*200;
        float[] buffer   = new float[bufferLength];
        int[]   inputs   = new int[]{count,1,200,200};
        float[]   imageBuffer = new float[200*200];
        
        INDArray ind = Nd4j.create(buffer, inputs);
        for(int i = 0; i < count; i++){
            int    index = i*(200*200);
            String  file = this.inputFiles.get(i);
            //System.out.println(":::: file : " + file);
            this.imageToBuffer(file, imageBuffer);
            int xy = 0;
            for(int x = 0; x < 200; x++){
                for(int y = 0; y < 200; y++){
                   ind.putScalar(i, 0, x, y, imageBuffer[xy]);
                   xy++;
                }
            }
            //System.arraycopy(imageBuffer, 0, buffer, index, 200*200);
        }
        
        return ind;
    }
    
    public INDArray getOutputArray(){

        int    count = inputFiles.size();
        int[] inputs = new int[]{count,numberOfClasses};
        int   bufferLength = count*numberOfClasses;
        float[] buffer = new float[bufferLength];
        
        for(int i = 0; i < count; i++){
            String filename = inputFiles.get(i);
            int startIndex = i*numberOfClasses;
            int         id = getImageClass(filename);
            for(int j = 0; j < numberOfClasses; j++){
                if(j==id) buffer[startIndex + j] = 1.0f;
                else buffer[startIndex + j] = 0.0f;
            }
        }
        return Nd4j.create(buffer, inputs);
    }
    
    public int getDataSetCount(){
        return 1;
    }
    
    
    public static class Filewalker {
        
        List<String> filesList = new ArrayList<String>();
        private String  extension = "";
        private boolean filterExtensions = false;
        
        public Filewalker(){
            
        }
        
        
        public void setFilterExtension(String ext){
            extension = ext;
            filterExtensions = true;
        }
        
        public void walk( String path ) {
            
            File root = new File( path );
            File[] list = root.listFiles();
            
            if (list == null) return;
            
            for ( File f : list ) {
                if ( f.isDirectory() ) {
                    walk( f.getAbsolutePath() );
                    //System.out.println( "Dir:" + f.getAbsoluteFile() );
                }
                else {
                   //System.out.println( "File:" + f.getAbsoluteFile() + " extension = " + f.getAbsoluteFile().toString().contains(extension)
                    //+ " extension = [" + extension +"]");
                    if(this.filterExtensions==false){
                        filesList.add(f.getAbsoluteFile().toString());
                    } else {
                        if(f.getAbsoluteFile().toString().contains(extension)==true){
                            filesList.add(f.getAbsoluteFile().toString());
                        }
                    }
                }
            }
        }
        
        public List<String> getList(){ return filesList;}
    }
    
    
    public static void main(String[] args){
                
        //Filewalker walker = new Filewalker();        
        //walker.walk("mldata");        
        //System.out.println("n files = " + walker.getList().size());
        
        String directory = "mldata";
        
        LineImageDataIterator iter = new LineImageDataIterator();

        iter.readDirectory(directory);
        
        String example = "/Users/gavalian/Work/Software/project-6a.0.0/Distribution/trackingML/clas12/training/mldata/.DS_Store extension";
        
        System.out.println( "PNG ? = " + example.contains("png"));
        //INDArray array = iter.getInputArray();
        
        
//        float[] buffer = new float[200*200];
//        iter.imageToBuffer(filename, buffer);
//        
//        int id = iter.getImageClass(filename);      
//        INDArray array = iter.getInputArray();
        //array.
        //System.out.println(" ID = " + id);
    }
}
