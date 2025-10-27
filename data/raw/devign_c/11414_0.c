void block_job_completed(BlockJob *job, int ret)

{

    BlockDriverState *bs = job->bs;



    assert(bs->job == job);

    job->cb(job->opaque, ret);

    bs->job = NULL;

    g_free(job);

    bdrv_set_in_use(bs, 0);

}
