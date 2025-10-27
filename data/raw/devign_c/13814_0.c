void qemu_aio_flush(void)

{

    AioHandler *node;

    int ret;



    do {

        ret = 0;



	/*

	 * If there are pending emulated aio start them now so flush

	 * will be able to return 1.

	 */

        qemu_aio_wait();



        QLIST_FOREACH(node, &aio_handlers, node) {

            if (node->io_flush) {

                ret |= node->io_flush(node->opaque);

            }

        }

    } while (qemu_bh_poll() || ret > 0);

}
