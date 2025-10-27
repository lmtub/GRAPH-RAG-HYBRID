void qmp_x_blockdev_set_iothread(const char *node_name, StrOrNull *iothread,

                                 Error **errp)

{

    AioContext *old_context;

    AioContext *new_context;

    BlockDriverState *bs;



    bs = bdrv_find_node(node_name);

    if (!bs) {

        error_setg(errp, "Cannot find node %s", node_name);

        return;

    }



    /* If we want to allow more extreme test scenarios this guard could be

     * removed.  For now it protects against accidents. */

    if (bdrv_has_blk(bs)) {

        error_setg(errp, "Node %s is in use", node_name);

        return;

    }



    if (iothread->type == QTYPE_QSTRING) {

        IOThread *obj = iothread_by_id(iothread->u.s);

        if (!obj) {

            error_setg(errp, "Cannot find iothread %s", iothread->u.s);

            return;

        }



        new_context = iothread_get_aio_context(obj);

    } else {

        new_context = qemu_get_aio_context();

    }



    old_context = bdrv_get_aio_context(bs);

    aio_context_acquire(old_context);



    bdrv_set_aio_context(bs, new_context);



    aio_context_release(old_context);

}
