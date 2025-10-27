static void mirror_complete(BlockJob *job, Error **errp)

{

    MirrorBlockJob *s = container_of(job, MirrorBlockJob, common);

    int ret;



    ret = bdrv_open_backing_file(s->target);

    if (ret < 0) {

        char backing_filename[PATH_MAX];

        bdrv_get_full_backing_filename(s->target, backing_filename,

                                       sizeof(backing_filename));

        error_set(errp, QERR_OPEN_FILE_FAILED, backing_filename);

        return;

    }

    if (!s->synced) {

        error_set(errp, QERR_BLOCK_JOB_NOT_READY, job->bs->device_name);

        return;

    }



    s->should_complete = true;

    block_job_resume(job);

}
