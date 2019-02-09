// $Id$
//
//    File: JEventProcessor_TrackMLextract.h
// Created: Fri Feb  1 07:09:07 EST 2019
// Creator: davidl (on Linux gluon47.jlab.org 2.6.32-642.3.1.el6.x86_64 x86_64)
//

#ifndef _JEventProcessor_TrackMLextract_
#define _JEventProcessor_TrackMLextract_

#include <mutex>
#include <fstream>

#include <JANA/JEventProcessor.h>
#include <TRACKING/DTrackTimeBased.h>


class JEventProcessor_TrackMLextract:public jana::JEventProcessor{
	public:
		JEventProcessor_TrackMLextract();
		~JEventProcessor_TrackMLextract();
		const char* className(void){return "JEventProcessor_TrackMLextract";}

		std::map<int, std::mutex> mtxs;    // key = geant_pid, value is mutex used to lock writing to corresponding element of ofs
		std::map<int, std::ofstream*> ofs; // key = geant_pid, value is track_parms.csv output file stream pointer
		std::map<int, std::ofstream*> bfs; // key = geant_pid, value is buffer output file stream pointer
		std::vector<int> cdc_nwires;       //< Number of wires for each CDC layer

		void WriteTrack(const DTrackTimeBased *tbt, uint32_t trackid, uint64_t eventnumber);

	private:
		jerror_t init(void);						///< Called once at program start.
		jerror_t brun(jana::JEventLoop *eventLoop, int32_t runnumber);	///< Called everytime a new run number is detected.
		jerror_t evnt(jana::JEventLoop *eventLoop, uint64_t eventnumber);	///< Called every event.
		jerror_t erun(void);						///< Called everytime run number changes, provided brun has been called.
		jerror_t fini(void);						///< Called after last event of last event source has been processed.
};

#endif // _JEventProcessor_TrackMLextract_

