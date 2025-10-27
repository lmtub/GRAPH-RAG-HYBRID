static void qemu_aio_wait_all(void)

{

    while (aio_poll(ctx, true)) {

        /* Do nothing */

    }

}
