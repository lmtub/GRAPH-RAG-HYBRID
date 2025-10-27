void backup_start(BlockDriverState *bs, BlockDriverState *target,

                  int64_t speed,

                  BlockdevOnError on_source_error,

                  BlockdevOnError on_target_error,

                  BlockDriverCompletionFunc *cb, void *opaque,

                  Error **errp)

{

    int64_t len;



    assert(bs);

    assert(target);

    assert(cb);



    if ((on_source_error == BLOCKDEV_ON_ERROR_STOP ||

         on_source_error == BLOCKDEV_ON_ERROR_ENOSPC) &&

        !bdrv_iostatus_is_enabled(bs)) {

        error_set(errp, QERR_INVALID_PARAMETER, "on-source-error");

        return;

    }



    len = bdrv_getlength(bs);

    if (len < 0) {

        error_setg_errno(errp, -len, "unable to get length for '%s'",

                         bdrv_get_device_name(bs));

        return;

    }



    BackupBlockJob *job = block_job_create(&backup_job_type, bs, speed,

                                           cb, opaque, errp);

    if (!job) {

        return;

    }



    job->on_source_error = on_source_error;

    job->on_target_error = on_target_error;

    job->target = target;

    job->common.len = len;

    job->common.co = qemu_coroutine_create(backup_run);

    qemu_coroutine_enter(job->common.co, job);

}
