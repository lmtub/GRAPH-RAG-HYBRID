void qmp_block_job_cancel(const char *device,

                          bool has_force, bool force, Error **errp)

{

    BlockJob *job = find_block_job(device);



    if (!has_force) {

        force = false;

    }



    if (!job) {

        error_set(errp, QERR_BLOCK_JOB_NOT_ACTIVE, device);

        return;

    }

    if (job->paused && !force) {

        error_setg(errp, "The block job for device '%s' is currently paused",

                   device);

        return;

    }



    trace_qmp_block_job_cancel(job);

    block_job_cancel(job);

}
