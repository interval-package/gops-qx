from gops.env.env_gen_ocp.resources.idsim_model.lasvsim_env_qianxing import LasvsimEnv
import numpy as np
import time
kwargs = {
    'token': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjQsIm9pZCI6MTAxLCJuYW1lIjoi6YK55paH5L-KIiwiaWRlbnRpdHkiOiJub3JtYWwiLCJwZXJtaXNzaW9ucyI6W10sImlzcyI6InVzZXIiLCJzdWIiOiJMYXNWU2ltIiwiZXhwIjoxNzE1NTg3NjU0LCJuYmYiOjE3MTQ5ODI4NTQsImlhdCI6MTcxNDk4Mjg1NCwianRpIjoiNCJ9.wVugIWCKsNPjmW5rAXGbD_jTWZETyomZpySbiaeht38',
    'task_id': 6120,
    'record_id': 10235,
    'b_surr':True
}

env = LasvsimEnv(**kwargs)
obs, info = env.reset()



for i in range(10000):
    start_time = time.time()
    action = env.action_space.sample()
    # action = np.array([1,1])  # Use this line if you want to test with a fixed action
    obs, reward, done, done, info = env.step(action)
    if done:
        env.reset()

    end_time = time.time()
    total_time = end_time - start_time
    print(f'Total time for step: {total_time} seconds')

    if i % 1000 == 0:
        print("-"*50+"step:"+str(i)+"-"*50)
        print("obs:{obs}, \n reward:{reward}, \n done:{done}, \n info:{info}, \n action:{action}".format(obs=obs, reward=reward, done=done, info=info, action=action))
    if done:
        print(f"state.msg: {env.stepResult.state.msg}")
        env.reset()
