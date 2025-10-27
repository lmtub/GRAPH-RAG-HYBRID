static void spin_reset(void *opaque)

{

    SpinState *s = opaque;

    int i;



    for (i = 0; i < MAX_CPUS; i++) {

        SpinInfo *info = &s->spin[i];



        info->pir = i;

        info->r3 = i;

        info->addr = 1;

    }

}
