static void aio_rfifolock_cb(void *opaque)

{

    /* Kick owner thread in case they are blocked in aio_poll() */

    aio_notify(opaque);

}
