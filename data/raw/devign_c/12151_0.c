static void qemu_aio_wait_all(void)

{

    while (qemu_aio_wait()) {

        /* Do nothing */

    }

}
