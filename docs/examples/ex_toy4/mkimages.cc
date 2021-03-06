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
// All tracks come from a vertex that is randomly selected
// from a 15cm target region centered 30cm upstream of 
// the first chamber's wire plane.
//
// The minimum distance from the most downstream vertex
// (z=7.5cm) to the last wire plane is (285-7.5 = 277.75cm).
// This gives a maximum angle for a track that hits
// all planes as being about 10.2 degrees. For tracks coming
// from the upstream target end, the maximum angle is 9.7
// degrees. The track angles are therefore limited to
// +/-9.7 degrees.
//
// Note that this does restrict the wires that can be hit 
// in the upstream chambers. The first chamber for example
// will only have the central 10 wires hit.
//
// A random hit efficiency may be imposed by setting the
// global hit_efficiency parameter. Its default setting is 
// 100% efficiency since they may facilitate a quicker
// initial training of the network. (This has not been
// benchmarked).
//
// Only the wire closest to the intesection point of the
// track to the wire plane is hit. (i.e. no double hits)



#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <zlib.h>
#include <stdint.h>
#include <sys/stat.h>

#include <fstream>
#include <iostream>
#include <vector>
#include <random>
#include <sstream>
#include <iomanip>
using namespace std;


double hit_efficiency = 1.0; // per pixel efficiency
bool HIT_ONLY = false; // true=no drift distance  false=include drift distance
uint8_t HIT_COLOR = 0xff;
uint8_t OFF_COLOR = 0x00;

//--------------------------------------
// MakeDataset
//--------------------------------------
void MakeDataset(string dirname, int Nevents)
{
	mkdir( dirname.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH );

	string imgfname = dirname+"/images.raw.gz";
	auto gzf = gzopen( imgfname.c_str(), "wb" );
	ofstream ofs( (dirname+"/track_parms.csv").c_str() );

	// Labels file header
	ofs << "filename,phi,z,phi_calc,phi_regression,sigma_regression" << endl;

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

	// Setup random number generator for phi
	std::random_device rd;  //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<> phi_dis(-10.0, 10.0); // degrees
	std::uniform_real_distribution<> x_dis(-7.5, 7.5); // cm
	std::uniform_real_distribution<> n_dis(0.0, 1.0);
	std::normal_distribution<> dd_dis(0.0, 0.0100); // gaussian with 100 micron sigma

	// Randomly shift each layer in y by a number between 3 and 97 
	// (generate these only on firsat call so all set use the same!)
	static vector<int> shifts;
	if( shifts.empty() ){
		//for(auto x : xplane) shifts.push_back( (int)(3.0 + 94.0*n_dis(gen)) );
		for(auto x : xplane) shifts.push_back( 0 );
		cout << "shifts: ";
		for( auto s : shifts ) cout << s << ", ";
		cout << endl;
	}

	// Loop over events	
	for(int ievent=0;ievent<Nevents; ievent++){
		
		// select random phi and z-offset
		auto phi = phi_dis(gen);
		auto x_vertex = x_dis(gen);

		// initialize everything as white
		memset(buff, OFF_COLOR, width*height);
		
		// Find track intersection point with each plane and set pixels.
		// We also calculate the sums needed for a linear regression of
		// the measurement points (wire positions) so we can calculate
		// slope and its uncertainty.
		double m = tan(phi/57.29578);
		double wsum = 0.0;
		double nsum = 0.0;
		double S=0.0, Sx=0.0, Sy=0.0, Sxx=0.0, Sxy=0.0;
		for(int igplane=0; igplane<xplane.size(); igplane++){ // igplane=0-35
			double x = xplane[igplane];
			double y = m*(x - x_vertex);
			int icol = igplane;          // 0-35
			int irow = floor(50.0 + y);  // 0-100
			
			// shift planes to break obvious patterns
			int shift = shifts[igplane];
			int irow_shifted = (irow+shift)%height;
			
			if( (icol>=width ) || (icol<0) ) break;
			if( (irow>=height) || (irow<0) ) break;
			if( n_dis(gen) > hit_efficiency ) continue;
			
			// Determine drift distance to wire and smear it by 100 microns.
			// We actualy just use the distance from the wire in the wire plane.
			// Here we use 150 color units = 5mm (the maximum drift distance)
			double y_bin_center = floor(y) + 0.5;  // wire position
			double drift_dist = fabs(y - y_bin_center + dd_dis(gen));
			if( HIT_ONLY ) drift_dist = 0.0;
			auto color = drift_dist*150.0/0.5; // 150 units of color = 0.5cm
			if( color>150.0 ) color = 150.0;
			if( HIT_COLOR == 0xff ) color = 255.0 - color;

			row_pointrs[irow_shifted][igplane] = (uint8_t)color;
			
			// For weighted average
			double dy = 1.0/sqrt(12.0); // error on position of single wire measurement
			double dphi = x*dy/(x*x + y_bin_center*y_bin_center);
			double w = 1.0/(dphi*dphi);
			wsum += w;
			nsum += w*atan2(y_bin_center, x)*57.29578;
			
			// For linear regression
			w = 1.0/(dy*dy);
			S   += w;
			Sx  += w*x;
			Sy  += w*y_bin_center;
			Sxx += w*x*x;
			Sxy += w*x*y_bin_center;
		}
		double phi_calc = nsum/wsum; // calculated phi from weighted average
		
		// calculated phi and uncertainty from linear regression
		double Delta = (S*Sxx) - (Sx*Sx);
		double slope = Sxy/Sxx;
		double sigma_slope = sqrt(fabs(1.0/Sxx));
		double phi_regression = atan(slope);
		double sigma_phi_regression = sigma_slope*pow( cos(phi_regression), 2.0 );
		phi_regression *= 57.29578;
		sigma_phi_regression *= 57.29578;

		gzwrite( gzf, buff, width*height );

		// Write to labels file
		char fname[256];
		sprintf( fname, "img%06d.png", ievent );
		ofs << fname << ", " << std::setprecision(9) << phi << ", " << x_vertex << ", " << phi_calc << ", " << phi_regression << ", " << sigma_phi_regression << endl;

		// Format is filename (not really used), true phi, calculated phi
		// The calculated phi is a weighted average for hit-based information.
		// It should represent the best possible answer for phi given the
		// information in the image. The difference between that and the true
		// phi gives the expected error.
		if( (ievent%100)== 0 ){
			cout << "   " << ievent << "/" << Nevents << " images written to " << imgfname << "      \r";
			cout.flush(); 
		}
	}
	
	cout << "   " << Nevents << "/" << Nevents << " images written to " << imgfname << "      " << endl;
	
	ofs.close();
	gzclose( gzf );
	delete[] buff;
}


//--------------------------------------
// main
//--------------------------------------
int main( int narg, char *argv[] )
{
	int Nimages = 500000;
	if(narg>1) Nimages = atoi(argv[1]);
	
	// Make all data sets (TEST and VALIDATION sets only have 1/10 events of TRAIN)
	MakeDataset( "TRAIN", Nimages );
	MakeDataset( "VALIDATION", Nimages/10 );
	MakeDataset( "TEST", Nimages/10 );

	return 0;
}
