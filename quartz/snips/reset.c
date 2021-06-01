#include <stdlib.h>
#include <string.h>
#include "time.h"
#include "reset.h"

static int numNeuronsPerCore = 1024;
static int logNumber = 0;

int tImgStart = 0;
int tImgEnd = 0;

extern int resetInterval;
extern int nCores;
extern int logInterval;
extern CoreId core_map[128];

int doReset(runState *RunState) {
    if(RunState->time_step % resetInterval == 0){
        return 1;
    }
    return 0;
}

void reset(runState *RunState) {
    NeuronCore* nc = NEURON_PTR(core_map[0]);

    CxState cxs = (CxState) {.U=0, .V=0};
    nx_fast_init_multicore(nc->cx_state, 
                           numNeuronsPerCore, 
                           sizeof(CxState), 
                           sizeof(CxState), 
                           &cxs,
                           &core_map[0],
                           nCores);     

    nx_fast_init_multicore(nc->dendrite_accum, 
                           numNeuronsPerCore * 8192 / 1024, 
                           sizeof(DendriteAccumEntry), 
                           sizeof(DendriteAccumEntry), 
                           0,
                           &core_map[0],
                           nCores);    
    
    logNumber += 1;
    if (logNumber % logInterval == 0){
        printf("QUARTZ: Done resetting cx_state and dendrite_accum. %d\n", RunState->time_step);
        tImgEnd = clock();
        printf("QUARTZ: Runtime per img = %dms, Avg. runtime per step = %dus time_step %d\n", (tImgEnd-tImgStart)/1000, (tImgEnd-tImgStart)/resetInterval, RunState->time_step);
        tImgStart = tImgEnd;
    }
}