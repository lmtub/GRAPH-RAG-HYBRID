BlockJobInfo *block_job_query(BlockJob *job)

{

    BlockJobInfo *info = g_new0(BlockJobInfo, 1);

    info->type      = g_strdup(BlockJobType_lookup[job->driver->job_type]);

    info->device    = g_strdup(bdrv_get_device_name(job->bs));

    info->len       = job->len;

    info->busy      = job->busy;

    info->paused    = job->paused;

    info->offset    = job->offset;

    info->speed     = job->speed;

    info->io_status = job->iostatus;

    info->ready     = job->ready;

    return info;

}
