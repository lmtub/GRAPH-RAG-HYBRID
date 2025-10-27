static int mux_chr_can_read(void *opaque)

{

    CharDriverState *chr = opaque;

    MuxDriver *d = chr->opaque;



    if ((d->prod - d->cons) < MUX_BUFFER_SIZE)

        return 1;

    if (d->chr_can_read[chr->focus])

        return d->chr_can_read[chr->focus](d->ext_opaque[chr->focus]);

    return 0;

}
