void qmp_x_blockdev_del(bool has_id, const char *id,

                        bool has_node_name, const char *node_name, Error **errp)

{

    AioContext *aio_context;

    BlockBackend *blk;

    BlockDriverState *bs;



    if (has_id && has_node_name) {

        error_setg(errp, "Only one of id and node-name must be specified");

        return;

    } else if (!has_id && !has_node_name) {

        error_setg(errp, "No block device specified");

        return;

    }



    if (has_id) {

        /* blk_by_name() never returns a BB that is not owned by the monitor */

        blk = blk_by_name(id);

        if (!blk) {

            error_setg(errp, "Cannot find block backend %s", id);

            return;

        }

        if (blk_legacy_dinfo(blk)) {

            error_setg(errp, "Deleting block backend added with drive-add"

                       " is not supported");

            return;

        }

        if (blk_get_refcnt(blk) > 1) {

            error_setg(errp, "Block backend %s is in use", id);

            return;

        }

        bs = blk_bs(blk);

        aio_context = blk_get_aio_context(blk);

    } else {

        blk = NULL;

        bs = bdrv_find_node(node_name);

        if (!bs) {

            error_setg(errp, "Cannot find node %s", node_name);

            return;

        }

        if (bdrv_has_blk(bs)) {

            error_setg(errp, "Node %s is in use by %s",

                       node_name, bdrv_get_parent_name(bs));

            return;

        }

        aio_context = bdrv_get_aio_context(bs);

    }



    aio_context_acquire(aio_context);



    if (bs) {

        if (bdrv_op_is_blocked(bs, BLOCK_OP_TYPE_DRIVE_DEL, errp)) {

            goto out;

        }



        if (!blk && !bs->monitor_list.tqe_prev) {

            error_setg(errp, "Node %s is not owned by the monitor",

                       bs->node_name);

            goto out;

        }



        if (bs->refcnt > 1) {

            error_setg(errp, "Block device %s is in use",

                       bdrv_get_device_or_node_name(bs));

            goto out;

        }

    }



    if (blk) {

        monitor_remove_blk(blk);

        blk_unref(blk);

    } else {

        QTAILQ_REMOVE(&monitor_bdrv_states, bs, monitor_list);

        bdrv_unref(bs);

    }



out:

    aio_context_release(aio_context);

}
