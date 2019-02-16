

void plot_results()
{

	auto t = new TTree("t","Results");
	t->ReadFile("phi_test.dat", "phi/F:phi_model");
	t->SetMarkerStyle(8);
	
	auto c1 = new TCanvas("c1","A canvas", 1200, 600);
	c1->Divide(2,1);
	
	c1->cd(1);
	t->Draw("phi_model:phi");

	c1->cd(2);
	gStyle->SetOptFit(1);
	t->Fit("gaus", "(phi_model-phi)*TMath::RadToDeg()");
}

