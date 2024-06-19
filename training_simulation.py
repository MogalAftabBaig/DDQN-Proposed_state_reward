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
PHASE_EW_GREEN = 4  # action 2 code 10 2-
PHASE_EW_YELLOW = 5 
PHASE_EWL_GREEN = 6  # action 3 code 11
PHASE_EWL_YELLOW = 7


class Simulation:
    def __init__(self, Model, Memory, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, delta, num_states, num_actions, training_epochs, max_length):
        self._Model = Model
        self._Memory = Memory
        self._TrafficGen = TrafficGen
        self._gamma = gamma
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._delta=delta
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_store = []
        self._cumulative_wait_store = []
        self._avg_queue_length_store = []
        self._avg_travel_time_store = []
        self._training_epochs = training_epochs
        self._max_length=max_length


    def run(self, episode, epsilon):
        """
        Runs an episode of simulation, then starts a training session
        """
        start_time = timeit.default_timer()

        # first, generate the route file for this simulation and set up sumo
        self._TrafficGen.generate_routefile_normal(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # inits
        self._step = 0
        self._waiting_times = {}
        self._emissions={}
        self._sum_neg_reward = 0
        self._sum_queue_length = 0
        self._sum_waiting_time = 0
        self._sum_travel_time = 0
        old_total_wait = 0
        old_queue=0
        max_queue=0
        old_state = []
        old_action = -1

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

            # saving the data into the memory
            if self._step != 0:
                self._Memory.add_sample((old_state, old_action, reward, current_state))

            # choose the light phase to activate, based on the current state of the intersection
            action = self._choose_action(current_state, epsilon)

            # if the chosen phase is different from the last phase, activate the yellow phase
            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            # execute the phase selected before
            #from here made the change
            traff=self._traffic(action)
            if traff>5:
                duration=10+self._delta
            elif traff<5:
                duration=10-self._delta
            self._green_duration=duration
            self._set_green_phase(action)
            self._simulate(self._green_duration)

            # saving variables for later & accumulate reward
            old_state = current_state
            old_action = action
            old_total_wait = current_total_wait
            old_queue=current_queue

            # saving only the meaningful reward to better see if the agent is behaving correctly
            if reward < 0:
                self._sum_neg_reward += reward

        
        self._save_episode_stats()
        print("Total reward:", self._sum_neg_reward, "- Epsilon:", round(epsilon, 2))
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        print("Training...")
        start_time = timeit.default_timer()
        for _ in range(self._training_epochs):
            self._replay()
        training_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time, training_time


    def _simulate(self, steps_todo):
        """
        Execute steps in sumo while gathering statistics
        """
        if (self._step + steps_todo) >= self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self._max_steps - self._step
        self.set_travel_time()
        tot_trav_time=self.collect_travel_time(self._step)
        self._sum_travel_time+=tot_trav_time
        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._step += 1 # update the step counter
            steps_todo -= 1
            queue_length = self._get_queue_length()
            self._sum_queue_length += queue_length
            self._sum_waiting_time += queue_length # 1 step while wating in queue means 1 second waited, for each car, therefore queue_lenght == waited_seconds


    def set_travel_time(self):
        incoming_roads = ["-h11", "-v11", "h12", "v12", "-h21", "-v12", "h22", "v13", "-h12", "-v21", "h13", "v22", "-h22", "-v22", "h23", "v23"]
        car_list=traci.vehicle.getIDList()
        for car_id in car_list:
            for edge_id in incoming_roads:
                traci.vehicle.setAdaptedTraveltime(car_id,edge_id,10)    

    def collect_travel_time(self, step):
        incoming_roads = ["-h11", "-v11", "h12", "v12", "-h21", "-v12", "h22", "v13", "-h12", "-v21", "h13", "v22", "-h22", "-v22", "h23", "v23"]
        car_list=traci.vehicle.getIDList()
        tot_trav_time=0
        for car_id in car_list:
            road_id=traci.vehicle.getRoadID(car_id)
            if road_id in incoming_roads:
                travel_time=traci.vehicle.getAdaptedTraveltime(car_id, step, road_id)
                tot_trav_time+=travel_time
        return tot_trav_time
            


    def _collect_waiting_times(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        incoming_roads = ["-h11", "-v11", "h12", "v12", "-h21", "-v12", "h22", "v13", "-h12", "-v21", "h13", "v22", "-h22", "-v22", "h23", "v23"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            #print the waiting times
            road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                self._waiting_times[car_id] = wait_time
            else:
                if car_id in self._waiting_times: # a car that was tracked has cleared the intersection
                    del self._waiting_times[car_id] 
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time
    
    def _collect_emissions(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        incoming_roads = ["-h11", "-v11", "h12", "v12", "-h21", "-v12", "h22", "v13", "-h12", "-v21", "h13", "v22", "-h22", "-v22", "h23", "v23"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            co=traci.vehicle.getCOEmission(car_id)
            co2=traci.vehicle.getCO2Emission(car_id)
            hc=traci.vehicle.getHCEmission(car_id)
            pmx=traci.vehicle.getPMxEmission(car_id)
            nox=traci.vehicle.getNOxEmission(car_id)
            avg_emissions=int((co+co2+hc+pmx+nox)/5000)
            road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                self._emissions[car_id] = avg_emissions
            else:
                if car_id in self._emissions: # a car that was tracked has cleared the intersection
                    del self._emissions[car_id] 
        total_emission = sum(self._emissions.values())
        return total_emission
    
    def _choose_action(self, state, epsilon):
        """
        Decide wheter to perform an explorative or exploitative action, according to an epsilon-greedy policy
        """
        if epsilon<0.02:
            epsilon=0.02
        if random.random() < epsilon:
            return random.randint(0, self._num_actions - 1) # random action
        else:
            return np.argmax(self._Model.predict_one(state)) # the best action given the current state


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
    #added this function for dynamic time purpose
    #to get amount of traffic 
    def _wait_time(self, action_number):
        res={}
        if action_number==0:
            lane_id=['-v11_0', '-v12_0', 'v13_0', 'v12_0', '-v21_0', '-v22_0', 'v23_0', 'v22_0']
            car_list = traci.vehicle.getIDList()
            for car_id in car_list:
                wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
                #print the waiting times
                road_id = traci.vehicle.getLaneID(car_id)  # get the road id where the car is located
                if road_id in lane_id:  # consider only the waiting times of cars in incoming roads
                    res[car_id] = wait_time
                else:
                    if car_id in res: # a car that was tracked has cleared the intersection
                        del res[car_id]
        if action_number==1:
            lane_id=['-v11_1', '-v12_1', 'v13_1', 'v12_1', '-v21_1', '-v22_1', 'v23_1', 'v22_1']
            car_list = traci.vehicle.getIDList()
            for car_id in car_list:
                wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
                #print the waiting times
                road_id = traci.vehicle.getLaneID(car_id)  # get the road id where the car is located
                if road_id in lane_id:  # consider only the waiting times of cars in incoming roads
                    res[car_id] = wait_time
                else:
                    if car_id in res: # a car that was tracked has cleared the intersection
                        del res[car_id]
        if action_number==2:
            lane_id=['-h11_0', '-h12_0', 'h13_0', 'h12_0', '-h21_0', '-h22_0', 'h23_0', 'h22_0']
            car_list = traci.vehicle.getIDList()
            for car_id in car_list:
                wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
                #print the waiting times
                road_id = traci.vehicle.getLaneID(car_id)  # get the road id where the car is located
                if road_id in lane_id:  # consider only the waiting times of cars in incoming roads
                    res[car_id] = wait_time
                else:
                    if car_id in res: # a car that was tracked has cleared the intersection
                        del res[car_id]
        if action_number==3:
            lane_id=['-h11_1', '-h12_1', 'h13_1', 'h12_1', '-h21_1', '-h22_1', 'h23_1', 'h22_1']
            car_list = traci.vehicle.getIDList()
            for car_id in car_list:
                wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
                #print the waiting times
                road_id = traci.vehicle.getLaneID(car_id)  # get the road id where the car is located
                if road_id in lane_id:  # consider only the waiting times of cars in incoming roads
                    res[car_id] = wait_time
                else:
                    if car_id in res: # a car that was tracked has cleared the intersection
                        del res[car_id]
        total_waiting_time = sum(res.values())
        return total_waiting_time
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
        length=max(halt_1,halt_2,halt_3,halt_4,halt_5,halt_6,halt_7,halt_8)
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



    def _replay(self):
        """
        Retrieve a group of samples from the memory and for each of them update the learning equation, then train
        """
        batch = self._Memory.get_samples(self._Model.batch_size)

        if len(batch) > 0:  # if the memory is full enough
            states = np.array([val[0] for val in batch])  # extract states from the batch
            next_states = np.array([val[3] for val in batch])  # extract next states from the batch

            # prediction
            q_s_a = self._Model.predict_batch(states)  # predict Q(state), for every sample
            q_s_a_d = self._Model.predict_batch(next_states)  # predict Q(next_state), for every sample
            #here made use of ddqn target network
            target_val=self._Model.target_predict_batch(next_states)

            # setup training arrays
            x = np.zeros((len(batch), self._num_states))
            y = np.zeros((len(batch), self._num_actions))

            for i, b in enumerate(batch):
                state, action, reward, _ = b[0], b[1], b[2], b[3]  # extract data from one sample
                current_q = q_s_a[i]  # get the Q(state) predicted before
                current_q[action] = reward + self._gamma * np.amax(target_val[i])  # update Q(state, action)
                x[i] = state
                y[i] = current_q  # Q(state) that includes the updated action value

            self._Model.train_batch(x, y)  # train the NN


    def _save_episode_stats(self):
        """
        Save the stats of the episode to plot the graphs at the end of the session
        """
        self._reward_store.append(self._sum_neg_reward)  # how much negative reward in this episode
        self._cumulative_wait_store.append(self._sum_waiting_time)  # total number of seconds waited by cars in this episode
        self._avg_queue_length_store.append(self._sum_queue_length / self._max_steps)  # average number of queued cars per step, in this episode
        self._avg_travel_time_store.append(self._sum_travel_time/ 200)


    @property
    def reward_store(self):
        return self._reward_store


    @property
    def cumulative_wait_store(self):
        return self._cumulative_wait_store


    @property
    def avg_queue_length_store(self):
        return self._avg_queue_length_store
    
    @property
    def avg_travel_time_store(self):
        return self._avg_travel_time_store
