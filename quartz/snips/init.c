#include <stdlib.h>
#include <string.h>
#include "init.h"

int resetInterval;
int enableReset;

static int channelID = -1;

void set_init_values(runState *s) {
    LOG("QUARTZ: Initializing...\n");
    if(channelID == -1) {
        channelID = getChannelID("init_channel");
        if(channelID == -1) {
          LOG("QUARTZ: Invalid channelID for nxinit\n");
        }
    }

    readChannel(channelID, &resetInterval, 1);
    //readChannel(channelID, &enableReset, 1);

    LOG("QUARTZ: reset interval=%d, log interval=100\n", resetInterval);
}

