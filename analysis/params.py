import numpy as np

class Params:
    def __init__(self):
        self.init_cost_params()

    def pl_fs(self, d):
        return 20*np.log10(d/1000) + 20*np.log10(self.f_c) + 92.45

    def pl_fs_custom(self, d, f, gal):
        return 20*np.log10(d/1000) + 20*np.log10(f) + 92.45


    def pl_wifi(self, d):
        #PL function taken from https://ieeexplore.ieee.org/document/995509
        pl = 38+25*np.log10(d)
        return  pl
    
    def pl_wifi_rural(self, d):
        #PL function taken from https://ieeexplore.ieee.org/document/995509
        pl = 21.8+33*np.log10(d)
        return  pl
    
    def set_radio(self, type):
        if type == '802.11ad_pro':
        #802.11ad technological values  
            self.speed_table = [
                (0, 12, 3200),
                (-59, 11, 2700),
                (-61, 10, 2200),
                (-63, 9,  1950),
                (-68, 8, 830),
                (-70, 7, 550),
                (-72, 4, 300),
                (-74, 3, 260),
                (-75, 0, 0)
            ]

            self.pt = 55 # (dBm) Maximum EIRP According to ETSI TR 102 555
            self.gr_m = 20 # (dBi) received gain for wave ap micro
            self.gr_d = 46 # (dBi) received gain for wave long range
            self.max_bx = 2200   #max ch capacity at mcs10
            self.mpant_bw = 90 #beamwidth of the mANTbox 19
            self.capex_costs['leaf_radio'] = 300
            self.capex_costs['mp_radio'] = 461
            self.f_c = 70 #GHz
            self.pl_function = self.pl_fs
            
            

        elif type == '802.11ac':
            self.speed_table = [
               (0,   13, 400),
               (-72, 12, 400),    #MCS Rx table derived by Mikrotik antbox19 datasheet (some values have been interpolated)
               (-75, 11, 360),
               (-77, 10, 300),
               (-83, 9, 270),
               (-86, 8, 240),
               (-90, 7, 180),
               (-92, 6, 120),
               (-94, 5, 90),
               (-95, 4, 60),
               (-96, 3, 30)] 
            
            self.pt = 30 # (dBm) Maximum EIRP According to ETSI TR 102 555
            self.gr_m = 19 # (dBi) received gain for wave ap micro
            self.gr_d = 27 # (dBi) received gain for wave long range
            self.max_bx = 300   #max ch capacity at mcs10
            self.mpant_bw = 120 #beamwidth of the mANTbox 19
            self.capex_costs['leaf_radio'] = 100
            self.capex_costs['mp_radio'] = 200
            self.f_c = 5.8 #GHz
            self.pl_function = self.pl_wifi



    def init_cost_params(self):
        self.capex_costs = {
            'gateway_deploy': 10000, #cost to deploy trellis + works + permits
            'gateway_router': 5000, # 
            'fiber_deploy': 6000, # per km (aerial)
            'relay_router': 500, # 
            'relay_deploy':  1000, # cost for trellis +  works 
            'leaf_deploy': 300, #cost for roof installation
        }

        #OpEx costs
        self.opex_costs = {
            'bw': 1680, #Euros per year for 1Gbps  [Cerdà 2020]
            'transport_10': 31200, # yearly price for transport of 10Gbps [xarxaoberta.cat]
            'transport_100': 55200,  # yearly price for transport of 100Gbps [xarxaoberta.cat]
            'planned_maintenance': 50, #euros per hour to repair
            'unplanned_maintenance': 200 #euros per hour to repair
        }


        self.power_consumption = {
            'leaf_radio': 26, # Watt
            'mp_radio': 24, # Watt
            'relay_router': 20, # Watt
            'gateway_router': 200, # Watt
        }

        self.cost_kw = 0.0003782 #Eur/W
        self.power_factor = 0.7

        #Reliability values
        self.mttf = {
            'router': 2e5, #hours [verbrugge 2006] 22y
            'radio': 1e5, #hours [mikrotik datasheet]  11yrs
        }

        self.mttr = {
            'router': 2, #hours [verbrugge 2006]
            'radio': 4, #hours [assumed]
        }