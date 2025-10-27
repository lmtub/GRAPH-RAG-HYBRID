static int analyze(const uint8_t *buf, int size, int packet_size, int *index,

                   int probe)

{

    int stat[TS_MAX_PACKET_SIZE];

    int stat_all = 0;

    int i;

    int best_score = 0;



    memset(stat, 0, packet_size * sizeof(*stat));



    for (i = 0; i < size - 3; i++) {

        if (buf[i] == 0x47 &&

            (!probe || (!(buf[i + 1] & 0x80) && buf[i + 3] != 0x47))) {

            int x = i % packet_size;

            stat[x]++;

            stat_all++;

            if (stat[x] > best_score) {

                best_score = stat[x];

                if (index)

                    *index = x;

            }

        }

    }



    return best_score - FFMAX(stat_all - 10*best_score, 0)/10;

}
