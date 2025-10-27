void block_job_user_resume(BlockJob *job)

{

    if (job && job->user_paused && job->pause_count > 0) {

        job->user_paused = false;

        block_job_iostatus_reset(job);

        block_job_resume(job);

    }

}
