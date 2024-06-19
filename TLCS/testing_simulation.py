import traci
import numpy as np
import random
import timeit
import os

# phase codes based on environment.net.xml
PHASE_NS_GREEN = 0  # action 0 code 00
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # action 1 code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action 2 code 10
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action 3 code 11
PHASE_EWL_YELLOW = 7


class Simulation:
    def __init__(self, Model, TrafficGen, sumo_cmd, max_steps, green_duration, delta, yellow_duration, num_states, num_actions):
        self._Model = Model
        self._TrafficGen = TrafficGen
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._delta=delta
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_episode = []
        self._queue_length_episode = []


    def run(self, episode):
        """
        Runs the testing simulation
        """
        start_time = timeit.default_timer()

        # first, generate the route file for this simulation and set up sumo
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # inits
        self._step = 0
        self._waiting_times = {}
        old_total_wait = 0
        old_queue=0
        max_queue=0
        old_action = -1 # dummy init
        max_length=0

        while self._step < self._max_steps:

            # get current state of the intersection
            current_state = self._get_state()

            # calculate reward of previous action: (change in cumulative waiting time between actions)
            # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
            current_total_wait = self._collect_waiting_times()
            current_queue=self._get_queue_length()
            max_queue=max(current_queue,max_queue)
            if max_queue!=0:
                reward=int(old_total_wait*(old_queue/max_queue)-current_total_wait*(current_queue/max_queue))
            else:
                reward=old_total_wait-current_total_wait

            # choose the light phase to activate, based on the current state of the intersection
            action = self._choose_action(current_state)
            print("Phase code of current action: ", action)
            # if the chosen phase is different from the last phase, activate the yellow phase
            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            # execute the phase selected before
            #dynamic time change
            traff=self._traffic(action)
            if traff>5:
                duration=10+self._delta
            elif traff<5:
                duration=10-self._delta
            self._green_duration=duration
            print("phase duration is: ", self._green_duration)
            self._set_green_phase(action)
            self._simulate(self._green_duration)

            # saving variables for later & accumulate reward
            old_action = action
            old_total_wait = current_total_wait

            self._reward_episode.append(reward)

        #print("Total reward:", np.sum(self._reward_episode))
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time


    def _simulate(self, steps_todo):
        """
        Proceed with the simulation in sumo
        """
        if (self._step + steps_todo) >= self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._step += 1 # update the step counter
            steps_todo -= 1
            queue_length = self._get_queue_length() 
            self._queue_length_episode.append(queue_length)


    def _collect_waiting_times(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        incoming_roads = ["-h11", "-v11", "h12", "v12", "-h21", "-v12", "h22", "v13", "-h12", "-v21", "h13", "v22", "-h22", "-v22", "h23", "v23"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                self._waiting_times[car_id] = wait_time
            else:
                if car_id in self._waiting_times: # a car that was tracked has cleared the intersection
                    del self._waiting_times[car_id] 
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time


    def _choose_action(self, state):
        """
        Pick the best action known based on the current state of the env
        """
        return np.argmax(self._Model.predict_one(state))


    def _set_yellow_phase(self, old_action):
        """
        Activate the correct yellow light combination in sumo
        """
        yellow_phase_code = old_action * 2 + 1 # obtain the yellow phase code, based on the old action (ref on environment.net.xml)
        traci.trafficlight.setPhase("1", yellow_phase_code)
        traci.trafficlight.setPhase("2", yellow_phase_code)
        traci.trafficlight.setPhase("5", yellow_phase_code)
        traci.trafficlight.setPhase("6", yellow_phase_code)


    def _set_green_phase(self, action_number):
        """
        Activate the correct green light combination in sumo
        """


        if action_number == 0:
            traci.trafficlight.setPhase("1", PHASE_NS_GREEN)
            traci.trafficlight.setPhase("2", PHASE_NS_GREEN)
            traci.trafficlight.setPhase("5", PHASE_NS_GREEN)
            traci.trafficlight.setPhase("6", PHASE_NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase("1", PHASE_NSL_GREEN)
            traci.trafficlight.setPhase("2", PHASE_NSL_GREEN)
            traci.trafficlight.setPhase("5", PHASE_NSL_GREEN)
            traci.trafficlight.setPhase("6", PHASE_NSL_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase("1", PHASE_EW_GREEN)
            traci.trafficlight.setPhase("2", PHASE_EW_GREEN)
            traci.trafficlight.setPhase("5", PHASE_EW_GREEN)
            traci.trafficlight.setPhase("6", PHASE_EW_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase("1", PHASE_EWL_GREEN)
            traci.trafficlight.setPhase("2", PHASE_EWL_GREEN)
            traci.trafficlight.setPhase("5", PHASE_EWL_GREEN)
            traci.trafficlight.setPhase("6", PHASE_EWL_GREEN)
    
    def _traffic(self, action_number):
        if action_number==0:
            halt_1 = traci.lane.getLastStepHaltingNumber("-v11_0")
            halt_2 = traci.lane.getLastStepHaltingNumber("-v12_0")
            halt_3 = traci.lane.getLastStepHaltingNumber("v13_0")
            halt_4 = traci.lane.getLastStepHaltingNumber("v12_0")
            halt_5 = traci.lane.getLastStepHaltingNumber("-v21_0")
            halt_6 = traci.lane.getLastStepHaltingNumber("-v22_0")
            halt_7 = traci.lane.getLastStepHaltingNumber("v23_0")
            halt_8 = traci.lane.getLastStepHaltingNumber("v22_0")
        elif action_number==1:
            halt_1 = traci.lane.getLastStepHaltingNumber("-v11_1")
            halt_2 = traci.lane.getLastStepHaltingNumber("-v12_1")
            halt_3 = traci.lane.getLastStepHaltingNumber("v13_1")
            halt_4 = traci.lane.getLastStepHaltingNumber("v12_1")
            halt_5 = traci.lane.getLastStepHaltingNumber("-v21_1")
            halt_6 = traci.lane.getLastStepHaltingNumber("-v22_1")
            halt_7 = traci.lane.getLastStepHaltingNumber("v23_1")
            halt_8 = traci.lane.getLastStepHaltingNumber("v22_1")
        elif action_number==2:
            halt_1 = traci.lane.getLastStepHaltingNumber("-h11_0")
            halt_2 = traci.lane.getLastStepHaltingNumber("-h12_0")
            halt_3 = traci.lane.getLastStepHaltingNumber("h13_0")
            halt_4 = traci.lane.getLastStepHaltingNumber("h12_0")
            halt_5 = traci.lane.getLastStepHaltingNumber("-h21_0")
            halt_6 = traci.lane.getLastStepHaltingNumber("-h22_0")
            halt_7 = traci.lane.getLastStepHaltingNumber("h23_0")
            halt_8 = traci.lane.getLastStepHaltingNumber("h22_0")
        elif action_number==3:
            halt_1 = traci.lane.getLastStepHaltingNumber("-h11_1")
            halt_2 = traci.lane.getLastStepHaltingNumber("-h12_1")
            halt_3 = traci.lane.getLastStepHaltingNumber("h13_1")
            halt_4 = traci.lane.getLastStepHaltingNumber("h12_1")
            halt_5 = traci.lane.getLastStepHaltingNumber("-h21_1")
            halt_6 = traci.lane.getLastStepHaltingNumber("-h22_1")
            halt_7 = traci.lane.getLastStepHaltingNumber("h23_1")
            halt_8 = traci.lane.getLastStepHaltingNumber("h22_1")
        length=(halt_1+halt_2+halt_3+halt_4+halt_5+halt_6+halt_7+halt_8)//8
        return length


    def _get_queue_length(self):
        """
        Retrieve the number of cars with speed = 0 in every incoming lane
        """
        halt_1 = traci.edge.getLastStepHaltingNumber("-h11")
        halt_2 = traci.edge.getLastStepHaltingNumber("-v11")
        halt_3 = traci.edge.getLastStepHaltingNumber("h12")
        halt_4 = traci.edge.getLastStepHaltingNumber("v12")
        halt_5 = traci.edge.getLastStepHaltingNumber("-h21")
        halt_6 = traci.edge.getLastStepHaltingNumber("-v12")
        halt_7 = traci.edge.getLastStepHaltingNumber("h22")
        halt_8 = traci.edge.getLastStepHaltingNumber("v13")
        halt_9 = traci.edge.getLastStepHaltingNumber("-h12")
        halt_10 = traci.edge.getLastStepHaltingNumber("-v21")
        halt_11 = traci.edge.getLastStepHaltingNumber("h13")
        halt_12 = traci.edge.getLastStepHaltingNumber("v22")
        halt_13 = traci.edge.getLastStepHaltingNumber("-h22")
        halt_14 = traci.edge.getLastStepHaltingNumber("-v22")
        halt_15 = traci.edge.getLastStepHaltingNumber("h23")
        halt_16 = traci.edge.getLastStepHaltingNumber("v23")
        
        queue_length = halt_1 + halt_2 + halt_3 + halt_4 + halt_5 + halt_6 + halt_7 + halt_8 + halt_9 + halt_10 + halt_11 + halt_12 + halt_13 + halt_14+ halt_15 + halt_16
        return queue_length

    def _get_state(self):
        """
        Retrieve the state of the intersection from sumo, in the form of cell occupancy
        """
        state_pos = np.zeros(self._num_states)
        state_speed=np.zeros(self._num_states)
        state=np.zeros(self._num_states)
        car_list = traci.vehicle.getIDList()
        max_speed=0
        min_speed=0

        for car_id in car_list:
            lane_pos = traci.vehicle.getLanePosition(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)
            lane_pos = 150 - lane_pos  # inversion of lane pos, so if the car is close to the traffic light -> lane_pos = 0 --- 750 = max len of a road

            # distance in meters from the traffic light -> mapping into cells
            if lane_pos < 7:
                lane_cell = 0
            elif lane_pos < 14:
                lane_cell = 1
            elif lane_pos < 21:
                lane_cell = 2
            elif lane_pos < 28:
                lane_cell = 3
            elif lane_pos < 35:
                lane_cell = 4
            elif lane_pos < 49:
                lane_cell = 5
            elif lane_pos < 63:
                lane_cell = 6
            elif lane_pos < 84:
                lane_cell = 7
            elif lane_pos < 122:
                lane_cell = 8
            elif lane_pos <=150:
                lane_cell = 9

            # finding the lane where the car is located 
            # x2TL_3 are the "turn left only" lanes
            if lane_id == "-h11_0":
                lane_group = 0
            elif lane_id == "-h11_1":
                lane_group = 1
            elif lane_id == "-v11_0":
                lane_group = 2
            elif lane_id == "-v11_1":
                lane_group = 3
            elif lane_id == "h12_0":
                lane_group = 4
            elif lane_id == "h12_1":
                lane_group = 5
            elif lane_id == "v12_0":
                lane_group = 6
            elif lane_id == "v12_1":
                lane_group = 7
            elif lane_id == "-h21_0":
                lane_group = 8
            elif lane_id == "-h21_1":
                lane_group = 9
            elif lane_id == "-v12_0":
                lane_group = 10
            elif lane_id == "-v12_1":
                lane_group = 11
            elif lane_id == "h22_0":
                lane_group = 12
            elif lane_id == "h22_1":
                lane_group = 13
            elif lane_id == "v13_0":
                lane_group = 14
            elif lane_id == "v13_1":
                lane_group = 15
            elif lane_id == "-h12_0":
                lane_group = 16
            elif lane_id == "-h12_1":
                lane_group = 17
            elif lane_id == "-v21_0":
                lane_group = 18
            elif lane_id == "-v21_1":
                lane_group = 19
            elif lane_id == "h13_0":
                lane_group = 20
            elif lane_id == "h13_1":
                lane_group = 21
            elif lane_id == "v22_0":
                lane_group = 22
            elif lane_id == "v22_1":
                lane_group = 23
            elif lane_id == "-h22_0":
                lane_group = 24
            elif lane_id == "-h22_1":
                lane_group = 25
            elif lane_id == "-v22_0":
                lane_group = 26
            elif lane_id == "-v22_1":
                lane_group = 27
            elif lane_id == "h23_0":
                lane_group = 28
            elif lane_id == "h23_1":
                lane_group = 29
            elif lane_id == "v23_0":
                lane_group = 30
            elif lane_id == "v23_1":
                lane_group = 31
            else:
                lane_group = -1

            if lane_group >= 1 and lane_group <= 31:
                car_position = int(str(lane_group) + str(lane_cell))  # composition of the two postion ID to create a number in interval 0-79
                valid_car = True
            elif lane_group == 0:
                car_position = lane_cell
                valid_car = True
            else:
                valid_car = False  # flag for not detecting cars crossing the intersection or driving away from it

            if valid_car:
                state_pos[car_position]=1
                speed=traci.vehicle.getSpeed(car_id)
                max_speed=max(max_speed,speed)
                if min_speed==0:
                    min_speed=speed
                else:
                    min_speed=min(min_speed,speed)
                if max_speed!=min_speed:
                    pos=(speed-min_speed)/(max_speed-min_speed)
                else:
                    pos=0
                state_speed[car_position] = pos # write the position of the car car_id in the state array in the form of "cell occupied"
                state[car_position]=state_pos[car_position]+state_speed[car_position]
        return state


    @property
    def queue_length_episode(self):
        return self._queue_length_episode


    @property
    def reward_episode(self):
        return self._reward_episode



