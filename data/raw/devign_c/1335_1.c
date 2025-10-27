static void commit_set_speed(BlockJob *job, int64_t speed, Error **errp)

{

    CommitBlockJob *s = container_of(job, CommitBlockJob, common);



    if (speed < 0) {

        error_setg(errp, QERR_INVALID_PARAMETER, "speed");

        return;

    }

    ratelimit_set_speed(&s->limit, speed / BDRV_SECTOR_SIZE, SLICE_TIME);

}
