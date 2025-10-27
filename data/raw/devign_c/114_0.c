static void mirror_complete(BlockJob *job, Error **errp)

{

    MirrorBlockJob *s = container_of(job, MirrorBlockJob, common);

    Error *local_err = NULL;

    int ret;



    ret = bdrv_open_backing_file(s->target, NULL, &local_err);

    if (ret < 0) {

        error_propagate(errp, local_err);

        return;

    }

    if (!s->synced) {

        error_setg(errp, QERR_BLOCK_JOB_NOT_READY,

                   bdrv_get_device_name(job->bs));

        return;

    }



    /* check the target bs is not blocked and block all operations on it */

    if (s->replaces) {

        AioContext *replace_aio_context;



        s->to_replace = check_to_replace_node(s->replaces, &local_err);

        if (!s->to_replace) {

            error_propagate(errp, local_err);

            return;

        }



        replace_aio_context = bdrv_get_aio_context(s->to_replace);

        aio_context_acquire(replace_aio_context);



        error_setg(&s->replace_blocker,

                   "block device is in use by block-job-complete");

        bdrv_op_block_all(s->to_replace, s->replace_blocker);

        bdrv_ref(s->to_replace);



        aio_context_release(replace_aio_context);

    }



    s->should_complete = true;

    block_job_enter(&s->common);

}
