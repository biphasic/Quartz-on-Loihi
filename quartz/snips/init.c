#include <stdlib.h>
#include <string.h>
#include "init.h"

int resetInterval;
int nCores;
int logInterval;
CoreId core_map[128];

static int channelID = -1;

void set_init_values(runState *s) {
    printf("QUARTZ: Initializing...\n");
    if(channelID == -1) {
        channelID = getChannelID("init_channel");
        if(channelID == -1) {
          printf("QUARTZ: Invalid channelID for init snip\n");
        }
    }

    readChannel(channelID, &resetInterval, 1);
    readChannel(channelID, &nCores, 1);
    logInterval = 100;
    
    // setup lookup table
    for(int ii=0; ii<128; ii++)
        core_map[ii] = nx_nth_coreid(ii);
    
    printf("QUARTZ: reset interval=%d, number of cores=%d, printf interval=%d\n", resetInterval, nCores, logInterval);
}

