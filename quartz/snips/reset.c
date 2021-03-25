#include <stdlib.h>
#include <string.h>
#include "time.h"
#include "reset.h"

static int numNeuronsPerCore = 1024;
static int NUM_Y_TILES = 5;
static int logNumber = 0;

int tImgStart = 0;
int tImgEnd = 0;

//extern int numCores;
extern int resetInterval;
extern int enableReset;

void logicalToPhysicalCoreId(int logicalId, CoreId *physicalId) {
    physicalId->p = logicalId % 4;
    physicalId->x = logicalId/(4*(NUM_Y_TILES-1));
    physicalId->y = (logicalId - physicalId->x*4*(NUM_Y_TILES-1))/4 + 1;
}

int doReset(runState *RunState) {
    if(RunState->time_step % resetInterval == 0){
        return 1;
    }
    return 0;
}

void reset(runState *RunState) {
    NeuronCore *nc;
    CoreId coreId;
    int numCores = 20;

//     int resetStart = clock();
    
    for(int i=0; i<numCores; i++) {
        logicalToPhysicalCoreId(i, &coreId);
        nc = NEURON_PTR(coreId);
        nx_fast_init64(nc->cx_state, numNeuronsPerCore, 0);//*(uint64_t*)&cxs);
    }
    
    for(int i=0; i<numCores; i++) {
        logicalToPhysicalCoreId(i, &coreId);
        nc = NEURON_PTR(coreId);
        nx_fast_init32(nc->dendrite_accum, 8192, 0);
    }

//     int resetFinish = clock();
    
    logNumber += 1;
    if (logNumber % 100 == 0){
        LOG("QUARTZ: Done resetting cx_state and dendrite_accum. %d\n", RunState->time_step);

//         LOG("QUARTZ: Reset duration = %dus\n", (resetFinish-resetStart)/1000);

        tImgEnd = clock();
        LOG("QUARTZ: Runtime per img = %dms, Avg. runtime per step = %dus time_step %d\n", (tImgEnd-tImgStart)/1000, (tImgEnd-tImgStart)/resetInterval, RunState->time_step);
        tImgStart = tImgEnd;
        logNumber = 0;
    }


}