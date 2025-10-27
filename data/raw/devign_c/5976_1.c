static void nfs_co_generic_bh_cb(void *opaque)

{

    NFSRPC *task = opaque;

    task->complete = 1;

    qemu_bh_delete(task->bh);

    qemu_coroutine_enter(task->co, NULL);

}
