static void stream_set_speed(BlockJob *job, int64_t speed, Error **errp)

{

    StreamBlockJob *s = container_of(job, StreamBlockJob, common);



    if (speed < 0) {

        error_set(errp, QERR_INVALID_PARAMETER, "speed");

        return;

    }

    ratelimit_set_speed(&s->limit, speed / BDRV_SECTOR_SIZE);

}
