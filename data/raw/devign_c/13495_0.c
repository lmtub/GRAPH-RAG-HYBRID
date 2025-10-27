void qmp_block_job_set_speed(const char *device, int64_t value, Error **errp)

{

    BlockJob *job = find_block_job(device);



    if (!job) {

        error_set(errp, QERR_DEVICE_NOT_ACTIVE, device);

        return;

    }



    if (block_job_set_speed(job, value) < 0) {

        error_set(errp, QERR_NOT_SUPPORTED);

    }

}
