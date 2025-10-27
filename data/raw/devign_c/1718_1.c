void qemu_co_queue_run_restart(Coroutine *co)

{

    Coroutine *next;



    trace_qemu_co_queue_run_restart(co);

    while ((next = QSIMPLEQ_FIRST(&co->co_queue_wakeup))) {

        QSIMPLEQ_REMOVE_HEAD(&co->co_queue_wakeup, co_queue_next);

        qemu_coroutine_enter(next, NULL);

    }

}
