static void block_job_ref(BlockJob *job)

{

    ++job->refcnt;

}
