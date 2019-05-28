/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.jlab.ml.tracking.clas;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;
import javax.imageio.ImageIO;

/**
 *
 * @author gavalian
 */
public class ImageProducer {
    
    private BufferedImage image = null;
    
    public ImageProducer(){}
    
    public void produceImage(List<TrackSegment> segments){
        image = createImage(segments);
    }
    
    public void saveImage(String name){
        try {
            // retrieve image            
            File outputfile = new File(name);
            ImageIO.write(image, "png", outputfile);
        } catch (IOException e) {
            System.out.println("oooops.....");
        }
    }
    
    public BufferedImage createImage(List<TrackSegment> segments){
        BufferedImage bi = new BufferedImage(112, 36, BufferedImage.TYPE_INT_RGB);
        for(int i = 0; i < segments.size(); i++){
            TrackSegment sg = segments.get(i);
            int   region = sg.getRegion();
            int imageRow = (region-1)*6;
            for(int wire = 0; wire < 112; wire++){
                for(int layer = 0; layer < 6; layer++){
                    double value = sg.getWire(layer, wire);
                    int      rgb = 0;
                    
                    if(value>0.1){
                        rgb = getRGBInt(255,255,255);
                    } else {
                        rgb = getRGBInt(  0,  0,  0);
                    }
                    bi.setRGB(wire, layer+imageRow, rgb);
                }
            }
        }
        return bi;
    }

            
    private int getRGBInt(int r, int g, int b){
        int result = ((0<<24) | (r<<16) | (g<<8) | b);
        return result;
    }
    
}
