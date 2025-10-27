static void stream_desc_store(struct Stream *s, hwaddr addr)

{

    struct SDesc *d = &s->desc;

    int i;



    /* Convert from host endianness into LE.  */

    d->buffer_address = cpu_to_le64(d->buffer_address);

    d->nxtdesc = cpu_to_le64(d->nxtdesc);

    d->control = cpu_to_le32(d->control);

    d->status = cpu_to_le32(d->status);

    for (i = 0; i < ARRAY_SIZE(d->app); i++) {

        d->app[i] = cpu_to_le32(d->app[i]);

    }

    cpu_physical_memory_write(addr, (void *) d, sizeof *d);

}
