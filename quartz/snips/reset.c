#include <stdlib.h>
#include <string.h>
#include "reset.h"

static int numNeuronsPerCore = 1024;
static int NUM_Y_TILES = 5;

//extern int numCores;
extern int resetInterval;
extern int enableReset;

void logicalToPhysicalCoreId(int logicalId, CoreId *physicalId) {
    physicalId->p = logicalId % 4;
    physicalId->x = logicalId/(4*(NUM_Y_TILES-1));
    physicalId->y = (logicalId - physicalId->x*4*(NUM_Y_TILES-1))/4 + 1;
}

int doReset(runState *s) {
    if(s->time_step%1600 == 0){
        return 1;
    }
    return 0;
}

void reset(runState *RunState) {
    NeuronCore *nc;
    CoreId coreId;
    int numCores = 128;

//     LOG("Resetting cores at runState->time_step=%d...\n", RunState->time_step);

    CxState cxs = (CxState) {.U=0, .V=0};
    for(int i=0; i<numCores; i++) {
        logicalToPhysicalCoreId(i, &coreId);
        nc = NEURON_PTR(coreId);
//        LOG("numCores=%d, core=%d, coreID=%d\n", numCores, i, coreId);
        nx_fast_init64(nc->cx_state, numNeuronsPerCore, *(uint64_t*)&cxs);
    }
        
    LOG("Done resetting cx_state. %d\n", RunState->time_step);
    
    for(int i=0; i<numCores; i++) {
        logicalToPhysicalCoreId(i, &coreId);
        nc = NEURON_PTR(coreId);
        nx_fast_init32(nc->dendrite_accum, 8192, 0);
    }

    LOG("Done resetting dendrite_accum. %d\n", RunState->time_step);
}