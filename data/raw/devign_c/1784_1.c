static void replication_start(ReplicationState *rs, ReplicationMode mode,

                              Error **errp)

{

    BlockDriverState *bs = rs->opaque;

    BDRVReplicationState *s;

    BlockDriverState *top_bs;

    int64_t active_length, hidden_length, disk_length;

    AioContext *aio_context;

    Error *local_err = NULL;



    aio_context = bdrv_get_aio_context(bs);

    aio_context_acquire(aio_context);

    s = bs->opaque;



    if (s->replication_state != BLOCK_REPLICATION_NONE) {

        error_setg(errp, "Block replication is running or done");

        aio_context_release(aio_context);

        return;

    }



    if (s->mode != mode) {

        error_setg(errp, "The parameter mode's value is invalid, needs %d,"

                   " but got %d", s->mode, mode);

        aio_context_release(aio_context);

        return;

    }



    switch (s->mode) {

    case REPLICATION_MODE_PRIMARY:

        break;

    case REPLICATION_MODE_SECONDARY:

        s->active_disk = bs->file;

        if (!s->active_disk || !s->active_disk->bs ||

                                    !s->active_disk->bs->backing) {

            error_setg(errp, "Active disk doesn't have backing file");

            aio_context_release(aio_context);

            return;

        }



        s->hidden_disk = s->active_disk->bs->backing;

        if (!s->hidden_disk->bs || !s->hidden_disk->bs->backing) {

            error_setg(errp, "Hidden disk doesn't have backing file");

            aio_context_release(aio_context);

            return;

        }



        s->secondary_disk = s->hidden_disk->bs->backing;

        if (!s->secondary_disk->bs || !bdrv_has_blk(s->secondary_disk->bs)) {

            error_setg(errp, "The secondary disk doesn't have block backend");

            aio_context_release(aio_context);

            return;

        }



        /* verify the length */

        active_length = bdrv_getlength(s->active_disk->bs);

        hidden_length = bdrv_getlength(s->hidden_disk->bs);

        disk_length = bdrv_getlength(s->secondary_disk->bs);

        if (active_length < 0 || hidden_length < 0 || disk_length < 0 ||

            active_length != hidden_length || hidden_length != disk_length) {

            error_setg(errp, "Active disk, hidden disk, secondary disk's length"

                       " are not the same");

            aio_context_release(aio_context);

            return;

        }



        if (!s->active_disk->bs->drv->bdrv_make_empty ||

            !s->hidden_disk->bs->drv->bdrv_make_empty) {

            error_setg(errp,

                       "Active disk or hidden disk doesn't support make_empty");

            aio_context_release(aio_context);

            return;

        }



        /* reopen the backing file in r/w mode */

        reopen_backing_file(bs, true, &local_err);

        if (local_err) {

            error_propagate(errp, local_err);

            aio_context_release(aio_context);

            return;

        }



        /* start backup job now */

        error_setg(&s->blocker,

                   "Block device is in use by internal backup job");



        top_bs = bdrv_lookup_bs(s->top_id, s->top_id, NULL);

        if (!top_bs || !bdrv_is_root_node(top_bs) ||

            !check_top_bs(top_bs, bs)) {

            error_setg(errp, "No top_bs or it is invalid");

            reopen_backing_file(bs, false, NULL);

            aio_context_release(aio_context);

            return;

        }

        bdrv_op_block_all(top_bs, s->blocker);

        bdrv_op_unblock(top_bs, BLOCK_OP_TYPE_DATAPLANE, s->blocker);



        backup_start(NULL, s->secondary_disk->bs, s->hidden_disk->bs, 0,

                     MIRROR_SYNC_MODE_NONE, NULL, false,

                     BLOCKDEV_ON_ERROR_REPORT, BLOCKDEV_ON_ERROR_REPORT,

                     BLOCK_JOB_INTERNAL, backup_job_completed, bs,

                     NULL, &local_err);

        if (local_err) {

            error_propagate(errp, local_err);

            backup_job_cleanup(bs);

            aio_context_release(aio_context);

            return;

        }

        break;

    default:

        aio_context_release(aio_context);

        abort();

    }



    s->replication_state = BLOCK_REPLICATION_RUNNING;



    if (s->mode == REPLICATION_MODE_SECONDARY) {

        secondary_do_checkpoint(s, errp);

    }



    s->error = 0;

    aio_context_release(aio_context);

}
