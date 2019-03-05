/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.jlab.ml.tracking.clas12;

import java.awt.BorderLayout;
import java.awt.FlowLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.List;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import org.jlab.groot.base.GStyle;
import org.jlab.groot.data.H1F;
import org.jlab.groot.data.H2F;
import org.jlab.groot.graphics.EmbeddedCanvas;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author gavalian
 */
public class TrackDataViewer extends JPanel implements ActionListener {
    
    EmbeddedCanvas canvas = new EmbeddedCanvas();
    Clas12DataLoader loader = new Clas12DataLoader();
    int currentTrack = 0;
    
    JLabel momentumLabel = null;
    JLabel thetaLabel = null;
    JLabel phiLabel = null;
    
    public TrackDataViewer(){
        super();
        initUI();
        canvas.divide(3, 5);
        GStyle.setPalette("kRainBow");
    }
    
    
    private void initUI(){
        this.setLayout(new BorderLayout());
        this.add(canvas,BorderLayout.CENTER);
        
        JPanel actionPanel = new JPanel();
        actionPanel.setLayout(new FlowLayout());
        JButton openFile = new JButton("Open");
        openFile.addActionListener(this);
        actionPanel.add(openFile);
        JButton nextButton = new JButton(">");
        nextButton.addActionListener(this);
        
        actionPanel.add(nextButton);
        
        JPanel infoPanel = new JPanel();
        
        momentumLabel = new JLabel("P = ");
        thetaLabel = new JLabel("Theta = ");
        phiLabel = new JLabel("Phi = ");
        
        infoPanel.setLayout(new FlowLayout());
        
        infoPanel.add(momentumLabel);
        infoPanel.add(thetaLabel);
        infoPanel.add(phiLabel);
        
        this.add(actionPanel,BorderLayout.PAGE_END);
        this.add(infoPanel,BorderLayout.PAGE_START);
    }
    
    public H2F getHistogram(TrackData data){
        List<TrackData> list = new ArrayList<TrackData>();
        list.add(data);
        H2F h = new H2F("H",112,0.0,112,36,0.0,36.0);
        INDArray ind = loader.getInputArray(list);
        for(int x = 0; x < 36; x++){
            for(int y = 0; y < 112; y++){
               double value = ind.getDouble(0,0,x,y);
               h.setBinContent(y,x, value);
            }
        }
        return h;
    }
    
    @Override
    public void actionPerformed(ActionEvent e) {
        if(e.getActionCommand().compareTo("Open")==0){
            loader.readFile("/Users/gavalian/Work/Software/project-6a.0.0/clas_004013_ML.hipo");
            currentTrack = 0;
        }
        
        if(e.getActionCommand().compareTo(">")==0){
            TrackData trk = loader.tracks().get(currentTrack); currentTrack++;
            H2F h = getHistogram(trk);
            //canvas.divide(1,2);
            //canvas.cd(0);
            canvas.getPad().setPalette("kRainBow");
            canvas.drawNext(h);
        }
    }
    
    public static void main(String[] args){
        JFrame frame = new JFrame();
        
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        
        TrackDataViewer viewer = new TrackDataViewer();
        frame.add(viewer);        
        frame.setSize(600,600);
        frame.setVisible(true);
    }

}
