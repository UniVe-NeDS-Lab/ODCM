import numpy as np

class Params:
    def __init__(self):
        self.init_cost_params()
        self.oxygen_loss_dict = {
                53: 1,
                54: 2.2,
                55: 4,
                56: 6.6,
                57: 9.7,
                58: 12.6,
                59: 14.6,
                60: 15,
                61: 14.6,
                62: 14.3,
                63: 10.5,
                64: 6.8,
                65: 3.9,
                66: 1.9,
                67: 1,
            }

    # def pl_fs(self, d):
    #     return 20*np.log10(d/1000) + 20*np.log10(self.f_c) + 92.45

    def pl_fs(self, d):
        return 20*np.log10(d/1000) + 20*np.log10(self.f_c) + 92.45 + self.get_oal(self.f_c)


    def pl_wifi(self, d):
        #PL function taken from https://ieeexplore.ieee.org/document/995509
        pl = 38+25*np.log10(d)
        return  pl
    
    def pl_wifi_rural(self, d):
        #PL function taken from https://ieeexplore.ieee.org/document/995509
        pl = 21.8+33*np.log10(d)
        return  pl
    
    def get_oal(self, frequency):
        if frequency <= 53 or frequency >= 67:
            return 0
        elif frequency in self.oxygen_loss_dict:
            return self.oxygen_loss_dict[frequency]
        else:
            lower_freq = max([f for f in self.oxygen_loss_dict if f < frequency])
            upper_freq = min([f for f in self.oxygen_loss_dict if f > frequency])
            
            lower_loss = self.oxygen_loss_dict[lower_freq]
            upper_loss = self.oxygen_loss_dict[upper_freq]
            
            # Linear interpolation
            interpolated_loss = lower_loss + (upper_loss - lower_loss) * (frequency - lower_freq) / (upper_freq - lower_freq)
            return interpolated_loss
    
    def set_radio(self, type):
        if type == '802.11ad_pro':
        #802.11ad technological values  for 2GHz BW
            # self.speed_table = [
            #     (0, 12, 3200),
            #     (-59, 11, 2700),
            #     (-61, 10, 2200),
            #     (-63, 9,  1950),
            #     (-68, 8, 830),
            #     (-70, 7, 550),
            #     (-72, 4, 300),
            #     (-74, 3, 260),
            #     (-75, 0, 0)
            # ]
        #802.11ad technological values  for 1GHz BW
            self.speed_table = [
                (0, 12, 1800),
                (-59, 11, 1500),
                (-61, 10, 1200),
                (-63, 9,  976),
                (-68, 8, 901),
                (-70, 7, 751),
                (-72, 4, 450),
                (-74, 3, 375),
                (-76, 1, 150),
                (-77, 0, 0)
            ]

            self.pt = 55 # (dBm) Maximum EIRP According to ETSI TR 102 555
            self.gr_m = 20 # (dBi) received gain for wave ap micro
            self.gr_d = 46 # (dBi) received gain for wave long range
            self.max_bx = 1200   #max ch capacity at mcs10
            self.mpant_bw = 90 #beamwidth of the mANTbox 19
            self.capex_costs['leaf_radio'] = np.array([180, 300])
            self.capex_costs['mp_radio'] = np.array([276, 461])
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
            self.capex_costs['leaf_radio'] = np.array([60, 100])
            self.capex_costs['mp_radio'] = np.array([120, 200])
            self.f_c = 5.8 #GHz
            self.pl_function = self.pl_wifi



    def init_cost_params(self):
        #Fiber cost from report 4.47$ to 9.25$ per feet.
        #https://fiberbroadband.org/wp-content/uploads/2024/01/Fiber-Deployment-Annual-Report-2023_FBA-and-Cartesian.pdf
        # Multiply by  3.28084 to get meter
        # 14.665$ to 30.034$ per meter
        # converto to Eur using PPP
        #1US$ equals 0.61 Eur
        #Eur 8.946 to 18.512


        self.capex_costs = {
            'gateway_deploy': np.array([10000, 10000]), #cost to deploy trellis + works + permits
            'gateway_router': np.array([3000, 5000]), # 
            'fiber_deploy': np.array([8946, 18512]),  #6000, # per km (aerial)
            'relay_router': np.array([300, 500]), # 
            'relay_deploy':  np.array([1000, 1000]), # cost for trellis +  works 
            'leaf_deploy': np.array([300, 300]), #cost for roof installation
        }

        #OpEx cost
        self.opex_costs = {
            'bw': np.array([1260, 1680]), #Euros per year for 1Gbps  [Cerdà 2020] https://dsg.ac.upc.edu/sites/default/files/dsg/Guifi_net_Economics.pdf
                                # range min is assuming 25% of traffic is peering max is assuming 0% is traffic
            'transport_10': np.array([27456, 27456]), # yearly price for transport of 10Gbps [xarxaoberta.cat from 2024 updated]
            'transport_100': np.array([55200, 55200]),  # yearly price for transport of 100Gbps [xarxaoberta.cat]
            'planned_maintenance': np.array([50,50]), #euros per hour to repair
            'unplanned_maintenance': np.array([200,200]) #euros per hour to repair
        }


        self.power_consumption = {
            'leaf_radio': 26, # Watt
            'mp_radio': 24, # Watt
            'relay_router': 20, # Watt
            'gateway_router': 200, # Watt
        }

        self.cost_kw = np.array([0.00018 ,0.00033]) #Eur/W
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