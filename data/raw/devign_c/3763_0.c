static void block_job_completed_txn_abort(BlockJob *job)

{

    AioContext *ctx;

    BlockJobTxn *txn = job->txn;

    BlockJob *other_job, *next;



    if (txn->aborting) {

        /*

         * We are cancelled by another job, which will handle everything.

         */

        return;

    }

    txn->aborting = true;

    /* We are the first failed job. Cancel other jobs. */

    QLIST_FOREACH(other_job, &txn->jobs, txn_list) {

        ctx = blk_get_aio_context(other_job->blk);

        aio_context_acquire(ctx);

    }

    QLIST_FOREACH(other_job, &txn->jobs, txn_list) {

        if (other_job == job || other_job->completed) {

            /* Other jobs are "effectively" cancelled by us, set the status for

             * them; this job, however, may or may not be cancelled, depending

             * on the caller, so leave it. */

            if (other_job != job) {

                block_job_cancel_async(other_job);

            }

            continue;

        }

        block_job_cancel_sync(other_job);

        assert(other_job->completed);

    }

    QLIST_FOREACH_SAFE(other_job, &txn->jobs, txn_list, next) {

        ctx = blk_get_aio_context(other_job->blk);

        block_job_completed_single(other_job);

        aio_context_release(ctx);

    }

}
