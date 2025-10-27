static void stream_desc_load(struct Stream *s, hwaddr addr)

{

    struct SDesc *d = &s->desc;

    int i;



    cpu_physical_memory_read(addr, (void *) d, sizeof *d);



    /* Convert from LE into host endianness.  */

    d->buffer_address = le64_to_cpu(d->buffer_address);

    d->nxtdesc = le64_to_cpu(d->nxtdesc);

    d->control = le32_to_cpu(d->control);

    d->status = le32_to_cpu(d->status);

    for (i = 0; i < ARRAY_SIZE(d->app); i++) {

        d->app[i] = le32_to_cpu(d->app[i]);

    }

}
