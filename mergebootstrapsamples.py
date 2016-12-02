import numpy as np

#cd('./mitosisData/bootstrapCollectedSamples')

tall = np.load('boot_round_1.npy')

t1 = np.load('boot_round_2.npy')
tall = np.concatenate((tall,t1), axis = 0 )

t1 = np.load('boot_round_3.npy')
tall = np.concatenate((tall,t1), axis = 0 )

t1 = np.load('boot_round_4.npy')
tall = np.concatenate((tall,t1), axis = 0 )

t1 = np.load('boot_round_5.npy')
tall = np.concatenate((tall,t1), axis = 0 )

t1 = np.load('boot_round_6.npy')
tall = np.concatenate((tall,t1), axis = 0 )

t1 = np.load('boot_round_7.npy')
tall = np.concatenate((tall,t1), axis = 0 )

t1 = np.load('boot_round_8.npy')
tall = np.concatenate((tall,t1), axis = 0 )

t1 = np.load('boot_round_9.npy')
tall = np.concatenate((tall,t1), axis = 0 )

t1 = np.load('boot_round_10.npy')
tall = np.concatenate((tall,t1), axis = 0 )

t1 = np.load('boot_round_11.npy')
tall = np.concatenate((tall,t1), axis = 0 )

t1 = np.save('boot_round_merged.npy',tall)

