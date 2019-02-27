
// Read in network_output.dat and make a 2D histo to
// plot the outputs.

void plot_outputs()
{
	vector<vector<double> > outputs;
	ifstream ifs("network_output.dat");
	char line[4096];
	while( ifs.getline( line, 4096 )){
		stringstream ss(line);
		vector<double> vals;
		//cout << "read: " << ss.str() << endl;
		while( !ss.eof() ){
			double val;
			ss >> val;
			//cout << "Found val: " << val <<  endl;
			if(!ss.eof()) vals.push_back(val);
		}
		outputs.push_back(vals);
	}
	
	auto Nouts = outputs[0].size();
	cout << "Found " << outputs.size() << " sample outputs with length " << Nouts << endl;
	


	TH2D *houts = new TH2D("houts", "Network Outputs;output class;sample", Nouts, 0.0, Nouts, outputs.size(), 0.0, outputs.size());
	int j=0;
	for(auto vals : outputs){
		int i=0;
		for(auto v : vals) houts->SetBinContent( i++, j, v);
		j++;
	}
	
	houts->SetStats(0);
	houts->Draw("colz");
}

