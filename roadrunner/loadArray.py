import numpy as np
import matplotlib.pyplot as plt
import statistics
import random
from rich import print
import json 
import time

jsonFilePath = "./turning_trajectory_library.json"

def selectGaitCyclePhase(trajOptFilePath):


    with open(trajOptFilePath) as trajOptData:
        data = json.load(trajOptData)
        #print(data) 

        #randomly choose b/t [1,21) to select gait velocity
        initVel = random.randrange(1,21,1)

        footGRFs = data[initVel]["input"][2]
        numOfNodesInAerial = 0
        if(footGRFs[0] == 0):
            for element in footGRFs:
                if(element == 0):
                    numOfNodesInAerial += 1
                else:
                    break

        gaitCycleFraction = 1 - (numOfNodesInAerial/50)
        
        a = data[5]["foot_loc"]
        b = data[5]["forward_vel"]
        print(a,b)
        return initVel, gaitCycleFraction
    


npzfile = np.load('./cassiePoses.npz', allow_pickle=True)

array = npzfile['x']

def requestCassiePose(initVel, gaitCyclePhase):
    """
    Given a gait velocity and a location for the starting point of the gait, return cassie qpos and qvel vectors
    initVel bounds: [0.1,2.0]
    gaitCycle bounds: [0,1]
    """

    velocityIndex = int(initVel - 1) #formula to convert desired commanded velocity to its corresponding array index 

    listLength = (len(array[velocityIndex][0])/2)-90
    


    #if left foot first
    #listIndex = int(gaitCyclePhase*listLength)
    #if right foot first
    listIndex = int(gaitCyclePhase*listLength) + int(len(array[velocityIndex][0])/2)

    qpos = array[velocityIndex][0][listIndex]["qpos"]
    qvel = array[velocityIndex][0][listIndex]["qvel"]
    print(array[velocityIndex][0][listIndex])
    return qpos.astype(float), qvel.astype(float)


    # for j in range(array.shape[1]):
        
    #     phaseAngle = array[velocityIndex][j][0]["phaseAngle"]
    #     pelv_Vel = array[velocityIndex][j][0]["pelvisVel"]
    #     left_foot_force = array[velocityIndex][j][0]["leftFootGRF_Z"]
    #     right_foot_force = array[velocityIndex][j][0]["rightFootGRF_Z"]
        
    #     if (phaseAngle in range( int(targetPhase - epsilon), int(targetPhase + epsilon))
    #         and left_foot_force == 0):

    #         qpos = array[velocityIndex][j][0]["qpos"]
    #         qvel = array[velocityIndex][j][0]["qvel"]
    #         return qpos.astype(float), qvel.astype(float)
        #targetPhaseAngle = ((-0.15/23)*i)+0.42
        #print(array[velocityIndex][j][-8:])
    

#qpos, qvel = requestCassiePose(0.2, 0.001)


initVel, gaitCycleFraction = selectGaitCyclePhase(jsonFilePath)
qpos, qvel = requestCassiePose(initVel, 1)
print(" starting cassie at {initVel} m/s ".format(initVel = initVel/10) + "qpos: {qpos} qvel: {qvel}".format(qpos = qpos, qvel = qvel) )
#print(qpos, qvel)

# cassie_logData[i,0] = "qpos"
# cassie_logData[i,1:36] = cassie_pose[0]
# cassie_logData[i,36] = "qvel"
# cassie_logData[i,37:69] = cassie_pose[1]
# cassie_logData[i,69] = "left foot force"
# cassie_logData[i,70] = cassie_pose[2]
# cassie_logData[i,71] = "right foot force"
# cassie_logData[i,72] = cassie_pose[3]
# cassie_logData[i,73] = "pelvis vel"
# cassie_logData[i,74] = cassie_pose[4]
# cassie_logData[i,75] = "phase angle"
# cassie_logData[i,76] = cassie_pose[5]