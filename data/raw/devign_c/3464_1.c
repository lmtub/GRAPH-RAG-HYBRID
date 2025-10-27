void block_job_cancel_sync(BlockJob *job)

{

    BlockDriverState *bs = job->bs;



    assert(bs->job == job);

    block_job_cancel(job);

    while (bs->job != NULL && bs->job->busy) {

        qemu_aio_wait();

    }

}
