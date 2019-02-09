
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <string>
#include <chrono>
#include <queue>
#include <thread>
#include <mutex>
using namespace std;

#include <png.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

uint32_t width = 720;
uint32_t height = 360;
uint32_t buff_size = width*height;

mutex buff_mtx;
queue<uint8_t*> buffers;

void CreatePNGfromBuffer( string fname, uint8_t *buff );

int main(int narg, char *argv[])
{


	// Open raw data file to get size
	string raw_fname = "images.raw";
	ifstream ifs_raw( raw_fname );
	ifs_raw.seekg(0, ios_base::end);
	auto len_tot = ifs_raw.tellg();
	ifs_raw.seekg(0, ios_base::beg);
	cout << "Input file: " << raw_fname << " opened with " << len_tot << " bytes (" << len_tot/1000000000 << " GB)" << endl;

	// Open index file
	string index_fname = "track_parms.csv";
	ifstream ifs( index_fname );

	// Get length of index file so we can estimate number of records
	// in order to pre-allocate the vector holding file names.
	ifs.seekg(0, ios_base::end);
	auto len_idx = ifs_raw.tellg();
	ifs.seekg(0, ios_base::beg);
	
	// Each line is roughly 231 bytes (from early example)
	std::vector<string>::size_type Nestimated_records = len_idx/231 + 1;
	vector<string> fnames;
	fnames.reserve(Nestimated_records);
	
	// Read in all file names from index
	char line[2048];
	ifs.getline( line, 2048 ); // read in and disregard header
	while( ifs.getline( line, 2048 ) ){
		// Find position of first comma denoting end of filename
		for(int i=0; i<2048; i++){
			if( line[i] == ','){
				line[i] = 0;
				break;
			}
			if( line[i] == 0 ) break;
		}
		fnames.push_back( line );
	}
	ifs.close();
	cout << "Found " << fnames.size() << " records in " << index_fname << endl;

	// Check that raw images file size matches number of records times buff_size 
	streampos expected_len = (streampos)buff_size * (streampos)fnames.size();
	if( expected_len != len_tot ){
		cout << "Expected size of raw images files does not match expectation!" << endl;
		cout << "  expected size: " << expected_len << endl;
		cout << "    actual size: " << len_tot << endl;
	}else{
		cout << "raw images file size matches expectation for " << width <<"x"<<height<<" image size" <<endl;
	}
	
	// Allocate buffers for images so we can use multiple threads
	int Nbuffers = 32;
	for(int i=0;i<Nbuffers; i++ ){
		auto buff = new uint8_t[ buff_size ];
		buffers.push(buff);
	}
	
	// Loop over images, creating PNG file for each
	int Nimages_written = 0;
	auto tstart = std::chrono::high_resolution_clock::now();
	for( auto fname : fnames ){
	
		// Get a buffer from the pool, waiting if neccessary 
		uint8_t *buff = nullptr;
		while( buff == nullptr ){
			if( buffers.empty() ) std::this_thread::sleep_for(std::chrono::microseconds(10));
			lock_guard<mutex> grd(buff_mtx);
			if( !buffers.empty() ) {
				buff = buffers.front();
				buffers.pop();
			}
		}

		// Read in image
		ifs_raw.read( (char*)buff, buff_size );
		
		// Create PNG file in separate thread
		thread thr( [=]{ CreatePNGfromBuffer( fname, buff );} );
		thr.detach();

// 		// Open output file
// 		FILE *fp = fopen(fname.c_str(), "wb");
// 		if(!fp){
// 			cerr << "Unable to open file " << fname << " for writing!" << endl;
// 			break;
// 		}
// 
// 		// Setup png structures
// 		png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING
// 			, png_voidp_NULL
// 			, png_error_ptr_NULL
// 			, png_error_ptr_NULL);
// 		if( !png_ptr ){
// 			cerr << "Error creating PNG write struct! " << endl;
// 			return -2;
// 		}
// 
// 		png_infop info_ptr = png_create_info_struct(png_ptr);
// 		if( !info_ptr ){
// 			png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
// 			cerr << "Unable to create PNG info struct!" << endl;
// 			return -3;
// 		}
// 
// 		if( setjmp( png_jmpbuf(png_ptr) ) ){
// 			png_destroy_write_struct(&png_ptr, &info_ptr);
// 			cerr << "Something went wrong and wayback was activated!" << endl;
// 			_exit(-1);
// 		}
// 		png_init_io(png_ptr, fp);
// 		png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
// 		png_write_info(png_ptr, info_ptr);
// 		png_write_image(png_ptr, row_pointrs.data());
// 		png_write_end(png_ptr, info_ptr);
// 		png_destroy_write_struct(&png_ptr, &info_ptr);
// 		fclose(fp);

		Nimages_written++;
		if( (Nimages_written%100) == 0 ){
			// Estimate time remaining
			auto now = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> tdiff = now - tstart;
			double rate = (double)Nimages_written/tdiff.count();
			double t_remaining = ((double)fnames.size() - (double)Nimages_written) / rate;
			double t_remaining_hours = floor(t_remaining/3600.0);
			double t_remaining_mins = floor((t_remaining-(t_remaining_hours*3600.0))/60.0);
			double t_remaining_secs = floor((t_remaining-(t_remaining_hours*3600.0) - (t_remaining_mins*60.0)));

			cout << "  " << Nimages_written << " / " << fnames.size() << " images written";
			cout << "  time remaining - ";
			if(t_remaining_hours>0.0) cout << (int)t_remaining_hours << "h ";
			if((t_remaining_mins>0.0) || (t_remaining_hours>0.0)) cout << (int)t_remaining_mins << "m ";
			cout << (int)t_remaining_secs << "s ";
			cout << "                              \r";
			cout.flush();
		}
		
	}
	
	// Wait for all threads to finish by virtue of all buffers being returned
	cout << "waiting for all threads to complete ..." << endl;
	while( buffers.size() != Nbuffers ) std::this_thread::sleep_for(std::chrono::microseconds(10));

	while( ! buffers.empty() ) { delete[] buffers.front(); buffers.pop(); }
	ifs_raw.close();
	
	cout << "Done" << endl;

	return 0;
}

//-----------------------------------
// CreatePNGfromBuffer
//-----------------------------------
void CreatePNGfromBuffer( string fname, uint8_t *buff )
{
	// Create row pointers
	vector<png_byte*> row_pointrs;
	for(png_uint_32 irow=0; irow<height; irow++){
		row_pointrs.push_back( &buff[irow*width] );
	}

	// Open output file
	FILE *fp = fopen(fname.c_str(), "wb");
	if(!fp){
		cerr << "Unable to open file " << fname << " for writing!" << endl;
		_exit(-1);
	}

	// Setup png structures
	png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING
		, png_voidp_NULL
		, png_error_ptr_NULL
		, png_error_ptr_NULL);
	if( !png_ptr ){
		cerr << "Error creating PNG write struct! " << endl;
		_exit(-2);
	}

	png_infop info_ptr = png_create_info_struct(png_ptr);
	if( !info_ptr ){
		png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
		cerr << "Unable to create PNG info struct!" << endl;
		_exit(-3);
	}

	if( setjmp( png_jmpbuf(png_ptr) ) ){
		png_destroy_write_struct(&png_ptr, &info_ptr);
		cerr << "Something went wrong and wayback was activated!" << endl;
		_exit(-1);
	}
	png_init_io(png_ptr, fp);
	png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
	png_write_info(png_ptr, info_ptr);
	png_write_image(png_ptr, row_pointrs.data());
	png_write_end(png_ptr, info_ptr);
	png_destroy_write_struct(&png_ptr, &info_ptr);
	fclose(fp);
	
	// Return buffer to pool
	lock_guard<mutex> grd(buff_mtx);
	buffers.push(buff);
}
