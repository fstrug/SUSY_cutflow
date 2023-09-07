#Time to complete: 639.5885846614838s elapsed, 328.6617329120636s elapsed, 367.95086455345154s elapsed, 61.23420310020447s, 50s, 48s
import time
import awkward as ak
import numpy as np
from coffea import processor
from coffea.nanoevents.methods import candidate
from coffea.nanoevents import NanoEventsFactory, BaseSchema
from coffea.analysis_tools import Weights, PackedSelection
import hist
import matplotlib.pyplot as plt

#Redefine to match ROOT definition
np.pi = 3.14159265358979323846

def delta_phi(a, b):
    return (a - b + np.pi) % np.pi - np.pi

class CutflowProcessor(processor.ProcessorABC):
    def __init__(self):
        self.lumi = 35.815165
    
    def process(self, events):
        print("Processing started")
        nevents = len(events)
        weights = Weights(nevents)
        ##add weight scale factors
        sign = np.sign(events['Stop0l_evtWeight'])
        weights.add("sign", weight = sign)
        weights.add("puWeight", weight = events["puWeight"])
        weights.add("PrefireWeight", weight = events["PrefireWeight"])
        weights.add("ISRWeight", weight = events["ISRWeight"])
        weights.add("nosplit_lumi", weight = self.lumi*np.ones(nevents))
        weights.add("Stop0l_ResTopWeight", weight = events["Stop0l_ResTopWeight"])
        weights.add("Stop0l_DeepAK8_SFWeight_fast", weight = events["Stop0l_DeepAK8_SFWeight_fast"])
        weights.add("trigger_efficiency", weight = events["Stop0l_trigger_eff_MET_loose_baseline"])

        #calculate muon_sf
        muons = ak.zip(
            {"pt": events.Muon_pt,
             "eta": events.Muon_eta,
             "is_mu": events.Muon_Stop0l,
             "muSF": events.Muon_LooseSF
             }
        )
        muons = muons[muons["is_mu"] & (muons["pt"] > 5) & (muons["eta"] < 2.5)]
        mu_SF = ak.prod(muons["muSF"], axis=1)
        weights.add("muon_SF_2", weight=mu_SF)

        #Calculate electron_SF
        electrons = ak.zip(
            {"pt": events.Electron_pt,
             "eta": events.Electron_eta,
             "is_e": events.Electron_Stop0l,
             "eSF": events.Electron_VetoSF
             }
        )
        electrons = electrons[electrons["is_e"] & (electrons["pt"] > 5) & (electrons["eta"] < 2.5)]
        e_sf = ak.prod(electrons["eSF"], axis=1)
        weights.add("electron_SF_2", weight=e_sf)
        
        #calculate B_SF
        B_SF = np.clip(events["BTagWeight"].to_numpy(), 0, 10)
        B_SF_fast = np.clip(events["BTagWeight_FS"].to_numpy(), 0 ,10)
        weights.add("B_SF", weight = B_SF)
        weights.add("B_SF_fast", weight = B_SF_fast)

        #Calculate SB_SF
        SB_Stop0l = events["SB_Stop0l"]
        SB_SF = ak.prod(ak.where(SB_Stop0l, events["SB_SF"], 1.0), axis=1)
        SB_fastSF = ak.prod(ak.where(SB_Stop0l, events["SB_fastSF"], 1.0), axis=1)
        weights.add("SB_Stop0l", weight = SB_SF)
        weights.add("SB_fastSF", weight = SB_fastSF)

        ##Selection Criteria
        #Baseline selection Criteria
        selection = PackedSelection()
        selection.add("Pass_EventFilter", events["Pass_EventFilter"])
        selection.add("Pass_JetID", events["Pass_JetID"])
        selection.add("Pass_CaloMETRatio", events["Pass_CaloMETRatio"])
        selection.add("Pass_MET", events["Pass_MET"])
        selection.add("Pass_Njets30", events["Pass_NJets30"])
        selection.add("Pass_LeptonVeto", events["Pass_ElecVeto"] & events["Pass_MuonVeto"])
        selection.add("Pass_TauVeto", events["Pass_TauVeto"])
        selection.add("Pass_IsoTrkVeto", events["Pass_IsoTrkVeto"])

        #High DM selection criteria
        selection.add("Pass_HT", events["Pass_HT"])
        selection.add("Stop0l_nJets >= 5", events["Stop0l_nJets"]>=5)
        selection.add("nbtagged >= 1", events["Stop0l_nbtags"]>=1)
        selection.add("Pass_dPhiMETHighDM", events["Pass_dPhiMETHighDM"])
        selection.add("Mtb > 175", events["Stop0l_Mtb"]>175)
        selection.add("Mtb < 175 and nJets >= 7", (events["Stop0l_Mtb"]<175) & (events["Stop0l_nJets"] >= 7))

        #low DM selection criteria
        selection.add("nTop=0, nW=0, and nresTop=0", (events["Stop0l_nTop"]==0) & (events["Stop0l_nW"] == 0) & (events["Stop0l_nResolved"]==0))
        selection.add("Mtb < 175", events["Stop0l_Mtb"] < 175)
        selection.add("Pass_dPhiMETLowDM", events["Pass_dPhiMETLowDM"])
        selection.add("Smet >= 10", (events["MET_pt"]/np.sqrt(events["Stop0l_HT"]))>=10.0)

        #ISR check
        FatJets_pts = ak.firsts(events["FatJet_pt"], axis=1)
        FatJets_eta = ak.firsts(events["FatJet_eta"], axis=1)
        FatJets_phi = ak.firsts(events["FatJet_phi"], axis=1)
        FatJets_btag = ak.firsts(events["FatJet_btagDeepB"], axis=1)
        Subjets_btag = events["SubJet_btagDeepB"]
        nbjets = ak.num(Subjets_btag, axis=1)

        #Subjet identification and tagging
        FatJets_subJetIdx1 = ak.firsts(events["FatJet_subJetIdx1"], axis=1)
        FatJets_subJetIdx1 = ak.mask(FatJets_subJetIdx1, FatJets_subJetIdx1 >= 0)
        FatJets_subJetIdx2 = ak.firsts(events["FatJet_subJetIdx2"], axis=1)
        FatJets_subJetIdx2 = ak.mask(FatJets_subJetIdx2, FatJets_subJetIdx2 >= 0)
        Subjet1_btag = ak.firsts(Subjets_btag[ak.singletons(FatJets_subJetIdx1)])
        Subjet2_btag = ak.firsts(Subjets_btag[ak.singletons(FatJets_subJetIdx2)])
        #For events with no subjets, say there is a proxy subjet that isn't btagged
        FatJets_subJetIdx1 = ak.fill_none(FatJets_subJetIdx1, 0)
        FatJets_subJetIdx2 = ak.fill_none(FatJets_subJetIdx2, 0)
        Subjet1_btag = ak.fill_none(Subjet1_btag, 0)
        Subjet2_btag = ak.fill_none(Subjet2_btag, 0)
        MET_phi = events["MET_phi"]
        working_point = 0.2217
        SAT_Pass_ISR = (
            (FatJets_pts >= 200.)
            & (abs(FatJets_eta) < 2.4)
            & (FatJets_btag < working_point)
            #subJet1
            & (~((FatJets_subJetIdx1 >= 0) & (FatJets_subJetIdx1 < nbjets) & (Subjet1_btag > working_point)))
            #subJet2
            & (~((FatJets_subJetIdx2 >= 0) & (FatJets_subJetIdx2 < nbjets) & (Subjet2_btag > working_point)))
            & (abs(delta_phi(FatJets_phi, MET_phi)) >= 2.0)
        )
        #Correct for events with no fatjets
        SAT_Pass_ISR = ak.fill_none(SAT_Pass_ISR, False)
        selection.add("SAT_Pass_ISR", SAT_Pass_ISR)

        
        ##Create histogram
        cutflow = hist.Hist.new.Reg(20, -0.5, 19.5, name=filename).Double()

        baseline_cuts = ["Pass_EventFilter", "Pass_JetID", "Pass_CaloMETRatio", "Pass_MET", "Pass_Njets30", "Pass_LeptonVeto", "Pass_TauVeto", "Pass_IsoTrkVeto"]
        Highdm_cuts = ["Pass_HT", "Stop0l_nJets >= 5", "nbtagged >= 1", "Pass_dPhiMETHighDM"]
        Highdm_final_cut = ["Mtb > 175", "Mtb < 175 and nJets >= 7"]
        Lowdm_cuts = ["nTop=0, nW=0, and nresTop=0", "Mtb < 175", "Pass_dPhiMETLowDM", "SAT_Pass_ISR", "Smet >= 10"]
        #Baseline Selection
        for i in range(len(baseline_cuts)):
            good_event = selection.all(*baseline_cuts[:i])
            bin_n = i
            cutflow.fill(bin_n*np.ones(len(events[good_event])), weight = weights.weight()[good_event])
        #Highdm Selection 
        for i in range(len(Highdm_cuts)):
            good_event = selection.all(*baseline_cuts) & selection.all(*Highdm_cuts[:i+1])
            bin_n = i+len(baseline_cuts)
            cutflow.fill(bin_n*np.ones(len(events[good_event])), weight = weights.weight()[good_event])
        #Different Highdm selection bins
        for i, cut in enumerate(Highdm_final_cut):
            good_event = selection.all(*baseline_cuts) & selection.all(*Highdm_cuts) & selection.all(cut)
            bin_n = i+len(baseline_cuts)+len(Highdm_cuts)
            cutflow.fill(bin_n*np.ones(len(events[good_event])), weight = weights.weight()[good_event])

        #Lowdm Selection
        for i in range(len(Lowdm_cuts)):
            good_event = selection.all(*baseline_cuts) & selection.all(*Lowdm_cuts[:i+1])
            bin_n = i + len(baseline_cuts) + len(Highdm_cuts) + len(Highdm_final_cut)
            cutflow.fill(bin_n*np.ones(len(events[good_event])), weight = weights.weight()[good_event])
        
        #Return cutflow with syntax for accumulator
        return({"data": {"cutflow": cutflow}})
    
    def postprocess(self, accumulator):
        pass

#Input ntuple
filename = "data/SMS_T2tt_mStop250_mLSP75_fastsim_2016_Skim_070650_37_082253_36.root"

#Set up processor and executor
futures_run = processor.Runner(
    executor = processor.IterativeExecutor(compression=None),
    schema = BaseSchema,
    chunksize = 100000,
    maxchunks = None
)

#Begin analysis
fileset = {"dataset": [filename]}
start_time = time.time()
cutflow = futures_run(fileset, "Events", processor_instance=CutflowProcessor())

#scale histogram
cutflow_hist = cutflow["data"]["cutflow"]
xs = 24.8
genfiltereff = 1
scale = 1 * xs * genfiltereff * 1000 / cutflow["data"]["nevents"]
cutflow_hist *= scale

end_time = time.time()
print("Done!")
print("{time}s elapsed".format(time=(end_time - start_time)))

#Saving histogram
plt.rcParams["figure.figsize"] = (10,10) #set figure size in in
plt.figure(0)
plt.gcf().subplots_adjust(bottom=0.30) #adjust margins
plt.gcf().subplots_adjust(left=0.15)
#scale histogram and plot it
cutflow_hist.plot()
x = np.arange(0,19)
labels = ["all", "Event Filter", "JetID", "CaloMETRatio", "MET>=250", "NJets>=2", "E/Mu Veto", "Tau Veto", "HT", "Njets >= 5", "Nb >= 1", "dPhi1234>0.5" ,"mTb > 175 and Njets >=5", "mTb < 175 and Njets >= 7", "0 top, w, resolved", "mTb < 175", "dphi jet met", "ISR pt > 200", "Smet >= 10"]
plt.xticks(x, labels, rotation=60) #set text at angle so it wont overlap
plt.ylabel("Events")
plt.xlabel("Cut")
plt.savefig("cutflow.png")