
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

int main( int narg, char *arv[])
{

	// Image parameters and buffer allocation
	int width  = 200;
	int height = 200;
	auto buff = new uint8_t[ width*height ];
	vector<uint8_t*> row_pointrs;
	for(uint32_t irow=0; irow<height; irow++){
		row_pointrs.push_back( &buff[irow*width] );
	}

	auto gzf = gzopen("images.raw.gz", "wb");
	ofstream ofs("track_parms.csv");

	// Labels file header
	ofs << "filename,event,trackid,q_over_pt,phi,tanl,D,z,cov00,cov01,cov02,cov03,cov04,cov11,cov12,cov13,cov13,cov14,cov22,cov23,cov24,cov33,cov34,cov44,pid" << endl;

	// Setup random number generator for phi and y-offset
	std::random_device rd;  //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<> phi_dis(-3.14, 3.14);
	std::uniform_real_distribution<> y_dis(height*0.20, height*0.80);
	std::uniform_real_distribution<> n_dis(0.0, 1.0);
	
	double hit_efficiency = 0.5; // per pixel efficiency
	
	int Nimages = 50000;
	for(int ievent=0;ievent<Nimages; ievent++){
		
		// select random phi and y-offset
		auto phi = phi_dis(gen);
		auto b   = y_dis(gen);

		// initialize everything as white
		memset(buff, 0xff, width*height);
		
		// Draw some points along line
		for(int i=0; i<width; i++){
			auto icol = int(i*cos(phi) + width/2);
			auto irow = int(i*sin(phi) + b);
			if( (icol>=width ) || (icol<0) ) break;
			if( (irow>=height) || (irow<0) ) break;
			if( n_dis(gen) < hit_efficiency ) continue;
			row_pointrs[irow][icol] = 0;
		}
/* 		for(int i=0; i<width; i++){
			auto icol = int(-i*cos(phi) + width/2);
			auto irow = int(-i*sin(phi) + b);
			if( (icol>=width ) || (icol<0) ) break;
			if( (irow>=height) || (irow<0) ) break;
			if( n_dis(gen) < hit_efficiency ) continue;
			row_pointrs[irow][icol] = 0;
		}
 */		gzwrite( gzf, buff, width*height );

		// Write to labels file
		char fname[256];
		sprintf( fname, "img%06d.png", ievent );
		stringstream ss;
		ss << fname << "," ;
		ss << ievent+1 << "," << 0 << ",";
		for(int i=0; i<5; i++) ss << (i==1 ? phi:0.0) << ",";            // state
		for(int i=0; i<5; i++) for(int j=i; j<5; j++) ss << 0.0 << "," ; // cov
		ss << 9;                                                         // PID
		ofs << ss.str() << endl;
		
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
