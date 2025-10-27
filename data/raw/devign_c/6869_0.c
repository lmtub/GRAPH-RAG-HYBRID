void block_job_cancel(BlockJob *job)

{

    job->cancelled = true;

    block_job_resume(job);

}
