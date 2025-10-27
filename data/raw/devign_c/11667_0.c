static int coroutine_fn nfs_co_flush(BlockDriverState *bs)

{

    NFSClient *client = bs->opaque;

    NFSRPC task;



    nfs_co_init_task(client, &task);



    if (nfs_fsync_async(client->context, client->fh, nfs_co_generic_cb,

                        &task) != 0) {

        return -ENOMEM;

    }



    while (!task.complete) {

        nfs_set_events(client);

        qemu_coroutine_yield();

    }



    return task.ret;

}
