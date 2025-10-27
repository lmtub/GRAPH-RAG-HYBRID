void DBDMA_register_channel(void *dbdma, int nchan, qemu_irq irq,

                            DBDMA_rw rw, DBDMA_flush flush,

                            void *opaque)

{

    DBDMAState *s = dbdma;

    DBDMA_channel *ch = &s->channels[nchan];



    DBDMA_DPRINTF("DBDMA_register_channel 0x%x\n", nchan);



    ch->irq = irq;

    ch->channel = nchan;

    ch->rw = rw;

    ch->flush = flush;

    ch->io.opaque = opaque;

    ch->io.channel = ch;

}
