void qmp_block_job_set_speed(const char *device, int64_t speed, Error **errp)

{

    BlockJob *job = find_block_job(device);



    if (!job) {

        error_set(errp, QERR_BLOCK_JOB_NOT_ACTIVE, device);

        return;

    }



    block_job_set_speed(job, speed, errp);

}
