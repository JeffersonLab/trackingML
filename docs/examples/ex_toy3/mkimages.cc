//
// This makes images of a toy model of a detector system
// where each pixel in the image represents a wire in a
// planar wire chamber. 
//
// The wire chambers are stacked in 6 groups of 6 giving
// 36 planes total (the x-dimension of the image). Each
// plane consists of 100 wires (the y-dimension).
//
// The wire spacing is 1cm and the distance between adjacent
// planes in a group is also 1 cm. The space between
// groups is 44cm thus the pattern repeats every 50cm.
//
// All tracks come from a vertex that is 30cm upstream of 
// the first chamber's wire plane.
//
// The distance from the vertex (z=0) to the last wire plane
// is 285cm. This gives a maximum angle for a track that hits
// all planes as being about 10 degrees. The track angles
// are therefore limited to +/-10 degrees.
//
// Note that this does restrict the wires that can be hit 
// in the upstream chambers. The first chamber for example
// will only have the central 10 wires hit.
//
// A random hit efficiency of 90% is imposed on each wire.
// Only the wire closest to the intesection point of the
// track to the wire plane is hit.



#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <zlib.h>
#include <stdint.h>

#include <fstream>
#include <iostream>
#include <vector>
#include <random>
#include <sstream>
using namespace std;

int main( int narg, char *argv[])
{

	// Image parameters and buffer allocation
	int width  = 36;
	int height = 100;
	auto buff = new uint8_t[ width*height ];
	vector<uint8_t*> row_pointrs;
	for(uint32_t irow=0; irow<height; irow++){
		row_pointrs.push_back( &buff[irow*width] );
	}
	
	// x-locations of planes relative to vertex
	vector<double> xplane;
	double x = 30.0; // position of first plane
	for(int igroup=0; igroup<6; igroup++){
		for(int ichamber=0; ichamber<6; ichamber++){
			xplane.push_back( x );
			x+=1.0;
		}
		x += 49.0; // already pushed to 1 past last plane of previous group
	}

	auto gzf = gzopen("images.raw.gz", "wb");
	ofstream ofs("track_parms.csv");

	// Labels file header
	ofs << "filename,phi" << endl;

	// Setup random number generator for phi
	std::random_device rd;  //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<> phi_dis(-10.0, 10.0); // degrees
	std::uniform_real_distribution<> n_dis(0.0, 1.0);
	
	double hit_efficiency = 0.90; // per pixel efficiency
	
	int Nimages = 50000;
	if(narg>1) Nimages = atoi(argv[1]);
	for(int ievent=0;ievent<Nimages; ievent++){
		
		// select random phi and y-offset
		auto phi = phi_dis(gen);

		// initialize everything as white
		memset(buff, 0xff, width*height);
		
		// Find track intersection point with each plane and
		// set pixels
		double m = tan(phi/57.29578);
		for(int igplane=0; igplane<xplane.size(); igplane++){ // igplane=0-35
			double y = m*xplane[igplane];
			int icol = igplane;          // 0-35
			int irow = floor(50.0 + y);  // 0-100
			if( (icol>=width ) || (icol<0) ) break;
			if( (irow>=height) || (irow<0) ) break;
			if( n_dis(gen) > hit_efficiency ) continue;
			row_pointrs[irow][igplane] = 0;
		}

		gzwrite( gzf, buff, width*height );

		// Write to labels file
		char fname[256];
		sprintf( fname, "img%06d.png", ievent );
		ofs << fname << "," << phi << endl;
		
		if( (ievent%100)== 0 ){
			cout << "   " << ievent << "/" << Nimages << " images written       \r";
			cout.flush(); 
		}
	}
	
	cout << endl;
	
	ofs.close();
	gzclose( gzf );

	return 0;
}
