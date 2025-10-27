void qemu_co_queue_restart_all(CoQueue *queue)

{

    while (qemu_co_queue_next(queue)) {

        /* Do nothing */

    }

}
