void qemu_chr_add_handlers(CharDriverState *s,

                           IOCanReadHandler *fd_can_read,

                           IOReadHandler *fd_read,

                           IOEventHandler *fd_event,

                           void *opaque)

{

    if (!opaque) {

        /* chr driver being released. */

        ++s->avail_connections;

    }

    s->chr_can_read = fd_can_read;

    s->chr_read = fd_read;

    s->chr_event = fd_event;

    s->handler_opaque = opaque;

    if (s->chr_update_read_handler)

        s->chr_update_read_handler(s);



    /* We're connecting to an already opened device, so let's make sure we

       also get the open event */

    if (s->opened) {

        qemu_chr_generic_open(s);

    }

}
