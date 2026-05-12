import pybullet as p
import pybullet_data
import time
import numpy as np
import cv2
import os
import csv
import random

# ==========================================================
# DATASET
# ==========================================================
DATASET_DIR = "nav_dataset"
IMG_DIR = os.path.join(DATASET_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)

csv_file = open(os.path.join(DATASET_DIR, "labels.csv"), "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["image", "kp", "kd"])

# ==========================================================
# VISION SYSTEM
# ==========================================================
class VisionSystem:
    def __init__(self, robot_id):
        self.robot_id = robot_id
        self.width = 320
        self.height = 240
        self.fov = 90

    def capture_camera(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3,3)

        cam_pos = np.array(pos) + np.array([0,0,0.28])
        target = cam_pos + rot.dot(np.array([1,0,0]))

        view = p.computeViewMatrix(cam_pos.tolist(), target.tolist(), [0,0,1])
        proj = p.computeProjectionMatrixFOV(self.fov,self.width/self.height,0.02,10)

        img = p.getCameraImage(self.width,self.height,view,proj,
                               renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb = np.reshape(img[2],(self.height,self.width,4)).astype(np.uint8)
        gray = cv2.cvtColor(rgb,cv2.COLOR_RGBA2GRAY)
        return gray

# ==========================================================
# TEACHER GAINS
# ==========================================================
def compute_optimal_gains(robot_pos, robot_vel, goal, obstacles):

    dist_goal=np.linalg.norm(np.array(robot_pos[:2])-np.array(goal[:2]))
    nearest=min(np.linalg.norm(np.array(robot_pos[:2])-np.array(o[:2])) for o in obstacles)
    speed=np.linalg.norm(robot_vel[:2])

    kp=1.5+3.0*np.exp(-nearest)+0.7*dist_goal
    kd=0.4+1.5*speed+1.2/(nearest+0.2)

    return np.clip(kp,0.5,6.0),np.clip(kd,0.1,4.0)

# ==========================================================
# LYAPUNOV NAVIGATION
# ==========================================================
def controller(pos,vel,yaw,target,obstacles):

    pos_error=np.array(target[:2])-np.array(pos[:2])
    dist=np.linalg.norm(pos_error)
    if dist<0.15:
        return 0,0,dist,yaw

    u_att=2.5*pos_error-1.2*np.array(vel[:2])

    u_rep=np.zeros(2)
    for obs in obstacles:
        dvec=np.array(pos[:2])-np.array(obs[:2])
        d=np.linalg.norm(dvec)
        if d<1.3:
            d=max(d,0.05)
            rep=(dvec/d)/(d*d)
            tan=np.array([-rep[1],rep[0]])
            u_rep+=2*rep+3*tan

    u=u_att+u_rep+np.array([0,0.15])

    theta=np.arctan2(u[1],u[0])
    err=np.arctan2(np.sin(theta-yaw),np.cos(theta-yaw))

    v=min(np.linalg.norm(u),1.2)
    omega=np.clip(2.0*err,-2,2)

    return v,omega,dist,theta

# ==========================================================
# RANDOM WORLD GENERATOR
# ==========================================================
def reset_episode(robot_id, obs_ids):

    start=[random.uniform(-1,1),random.uniform(-1,1),0.1]
    yaw=random.uniform(-np.pi,np.pi)
    orn=p.getQuaternionFromEuler([0,0,yaw])

    goal=[random.uniform(4,7),random.uniform(-2,2),0]

    p.resetBasePositionAndOrientation(robot_id,start,orn)
    p.resetBaseVelocity(robot_id,[0,0,0],[0,0,0])

    p.resetBasePositionAndOrientation(goal_id,goal,[0,0,0,1])

    phases=[]
    for i,oid in enumerate(obs_ids):
        phase=random.uniform(0,6.28)
        phases.append(phase)

    return goal,phases

# ==========================================================
# MAIN
# ==========================================================
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.8)
p.setTimeStep(1/240)

p.loadURDF("plane.urdf")

robot_id=p.createMultiBody(
    baseMass=1,
    baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX,halfExtents=[0.2,0.2,0.1],rgbaColor=[0,0,1,1]),
    baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX,halfExtents=[0.2,0.2,0.1]),
    basePosition=[0,0,0.1])

goal_id=p.createMultiBody(baseMass=0,
    baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE,radius=0.2,rgbaColor=[0,1,0,1]))

initial_obs=[[2,1.2,0.5],[2,-1.2,0.5],[3,0.6,0.5],[3,-0.6,0.5],[4,1.4,0.5],[4,-1.4,0.5]]

obs_ids=[]
for pos in initial_obs:
    oid=p.createMultiBody(baseMass=0,
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX,halfExtents=[0.25,0.25,0.5],rgbaColor=[1,0,0,1]),
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX,halfExtents=[0.25,0.25,0.5]),
        basePosition=pos)
    obs_ids.append(oid)

vision=VisionSystem(robot_id)

goal,phases=reset_episode(robot_id,obs_ids)

img_id=0
step=0

while True:

    # move obstacles
    cur_obs=[]
    for i,oid in enumerate(obs_ids):
        base=initial_obs[i]
        newy=base[1]+0.7*np.sin(step*0.01+phases[i])
        p.resetBasePositionAndOrientation(oid,[base[0],newy,base[2]],[0,0,0,1])
        cur_obs.append([base[0],newy,base[2]])

    pos,orn=p.getBasePositionAndOrientation(robot_id)
    vel,_=p.getBaseVelocity(robot_id)
    yaw=p.getEulerFromQuaternion(orn)[2]

    v,omega,dist,theta=controller(pos,vel,yaw,goal,cur_obs)
    p.resetBaseVelocity(robot_id,[v*np.cos(theta),v*np.sin(theta),0],[0,0,omega])

    # FOLLOW CAMERA
    p.resetDebugVisualizerCamera(cameraDistance=4,
                                 cameraYaw=0,
                                 cameraPitch=-60,
                                 cameraTargetPosition=pos)

    # DATASET
    if step%4==0:
        img=vision.capture_camera()
        kp,kd=compute_optimal_gains(pos,vel,goal,cur_obs)

        name=f"img_{img_id:07d}.png"
        cv2.imwrite(os.path.join(IMG_DIR,name),img)
        writer.writerow([name,kp,kd])
        print("saved",name,"kp",round(kp,2),"kd",round(kd,2))
        img_id+=1

    # NEW EPISODE
    if dist<0.2:
        goal,phases=reset_episode(robot_id,obs_ids)
        print("NEW EPISODE")

    p.stepSimulation()
    time.sleep(1/240)
    step+=1