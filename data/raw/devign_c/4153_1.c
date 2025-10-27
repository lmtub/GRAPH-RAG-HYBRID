static void coroutine_fn qed_co_pwrite_zeroes_cb(void *opaque, int ret)

{

    QEDWriteZeroesCB *cb = opaque;



    cb->done = true;

    cb->ret = ret;

    if (cb->co) {

        qemu_coroutine_enter(cb->co, NULL);

    }

}
