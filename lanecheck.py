import traci
from utils import set_sumo
sumocmd=set_sumo(False, 'chk.sumocfg', 500)
traci.start(sumocmd)
print(traci.vehicle.getLaneID("v_0"))
traci.close()