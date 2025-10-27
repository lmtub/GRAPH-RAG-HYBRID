static void pit_reset(void *opaque)

{

    PITState *pit = opaque;

    PITChannelState *s;

    int i;



    for(i = 0;i < 3; i++) {

        s = &pit->channels[i];

        s->mode = 3;

        s->gate = (i != 2);

        pit_load_count(s, 0);

    }

}
