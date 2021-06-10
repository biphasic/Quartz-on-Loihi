#include <stdlib.h>
#include <string.h>
#include "init.h"

int resetInterval;
int nCores;
int logInterval;
CoreId core_map[128];

static int channelID = -1;

void set_init_values(runState *s) {
    if(channelID == -1) {
        // this is a hacky and incredibly complicated way to concatenate a string 
        // and the logical_chip_id int because std::to_string is not available
        char channel_name_string[21];
        sprintf(channel_name_string, "init_channel_%d", get_logical_chip_id());
        printf("QUARTZ: Initializing and reading from channel name %s \n", channel_name_string);
        
        channelID = getChannelID(channel_name_string);
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

int get_logical_chip_id() {
    int chipid = -1;
    for (int ii=0; ii<nx_num_chips(); ii++)
        if (nx_nth_chipid(ii).id == nx_my_chipid().id) {
            chipid = ii;
            break;
    }
    return chipid;
}