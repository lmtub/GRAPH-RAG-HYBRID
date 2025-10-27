static void wait_for_aio(void)

{

    while (aio_poll(ctx, true)) {

        /* Do nothing */

    }

}
