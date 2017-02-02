import numpy as np

BOOTFOLDER='~/data/mitosisData/bootstrapCollectedSamples/'
BOOTFOLDER= '/home/btek/data/mitosisData/bootstrapCollectedSamples/'




ROOTFOLDER2 = '/home/btek/data/mitosisData/'





starti = 5;
endi=18;
tall = np.load(BOOTFOLDER+'boot_round_'+str(starti)+'.npy')
for i in range(starti+1,endi+1):
    t1 = np.load(BOOTFOLDER+'boot_round_'+str(i)+'.npy')
    tall = np.concatenate((tall,t1), axis = 0 )

    
import datetime 
now_t=datetime.datetime.now().isoformat()


np.save(BOOTFOLDER+'boot_round_merged_'+now_t+'.npy',tall)


# now load original files and merge
ORIGFILE = ROOTFOLDER2+'OriginalSampleX.npy'
ORIGTARGETCLASSFILE = ROOTFOLDER2+'OriginalSampleY.npy'

X = np.load(ORIGFILE)
Y = np.load(ORIGTARGETCLASSFILE)

X = np.concatenate((X, tall), axis=0)
Y= np.concatenate((Y, np.zeros(tall.shape[0])))

OUTFILE = ROOTFOLDER2+'OriginalSampleX_wb'
OUTTARGETCLASSFILE = ROOTFOLDER2+'OriginalSampleY_wb'

np.save(OUTFILE + now_t+'.npy',X)
np.save(OUTTARGETCLASSFILE + now_t+'.npy',Y)

