/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.jlab.ml.tracking.nn;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;

/**
 *
 * @author gavalian
 */
public class DataProducer {
    
    public BufferedImage createImage(int sizeX, int sizeY, double angle, 
            double efficiency){
        
        BufferedImage bi = new BufferedImage(sizeX, sizeY, BufferedImage.TYPE_INT_RGB);        
        int centerX = sizeX/2;
        int centerY = sizeY/2;
        
        
      
        for(int x = 0; x < sizeX; x++){
            for(int y = 0; y < sizeY; y++){
                bi.setRGB(x, y, getRGBInt(0,0,0));
            }
        }
            
        double radious = centerX;
        
        for(double r = 0; r < radious; r+=0.2){
            double xc = r*Math.cos(angle);
            double yc = r*Math.sin(angle);
            int    xp = (int) (centerX + xc);
            int    yp = (int) (centerY + yc);
            double seed = Math.random();
            //System.out.println(String.format(" x = %4d , yc = %4d", xp,yp));
            if(seed<efficiency)
                bi.setRGB(xp, yp, getRGBInt(255,255,255));
        }
        return bi;
    }
    
    public void saveImage(BufferedImage img, String name){
        try {
            // retrieve image            
            File outputfile = new File(name);
            ImageIO.write(img, "png", outputfile);
        } catch (IOException e) {
            System.out.println("oooops.....");
        }
    }
    
    public void saveImage(BufferedImage img, String dir, String name){
        File file = new File(dir);
        if(file.exists()==false){
            System.out.println("creating directory : " + dir);
            file.mkdir();
        }
        saveImage( img, dir + "/" + name);
    }
    
    private int getRGBInt(int r, int g, int b){
        int result = ((0<<24) | (r<<16) | (g<<8) | b);
        return result;
    }
    
    public String getImageNumberString(Integer number){
        int size = number.toString().length();
        StringBuilder str = new StringBuilder();
        int remainder = 12 - size;
        for(int i = 0; i < remainder; i++) str.append("0");
        str.append(number.toString());
        return str.toString();
    }        
    
    public void generateDataSet(String directory, int samples){
        
        for (int i = 0; i < samples; i++){
            Integer      number = (int) (Math.random()*180.0);            
            double   efficiency = 0.01 + Math.random()*0.5;
            double        angle = Math.toRadians(number);
            String     filename = "image_" + getImageNumberString(i) + ".png";
            String         path = directory + "/" + number.toString();
            BufferedImage    bi = this.createImage(200, 200, angle, efficiency);
            this.saveImage(bi, path, filename);
            System.out.println(path + "  -->  " + filename + " : " 
                    + angle + " deg -> " + angle*57.29);
            
        }
    }
    
    public static void main(String[] args){
        DataProducer producer = new DataProducer();
        BufferedImage bi = producer.createImage(200, 200, 1.57*2/4,0.1);
        producer.saveImage(bi, "dataImage.png");
        
        producer.generateDataSet("mldata", 1000);
        
    }
}
