static void DMA_run (void)

{

    struct dma_cont *d;

    int icont, ichan;

    int rearm = 0;

    static int running = 0;



    if (running) {

        rearm = 1;

        goto out;

    } else {

        running = 1;

    }



    d = dma_controllers;



    for (icont = 0; icont < 2; icont++, d++) {

        for (ichan = 0; ichan < 4; ichan++) {

            int mask;



            mask = 1 << ichan;



            if ((0 == (d->mask & mask)) && (0 != (d->status & (mask << 4)))) {

                channel_run (icont, ichan);

                rearm = 1;

            }

        }

    }



    running = 0;

out:

    if (rearm)

        qemu_bh_schedule_idle(dma_bh);

}
