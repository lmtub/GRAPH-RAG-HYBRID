static void coroutine_fn stream_run(void *opaque)

{

    StreamBlockJob *s = opaque;

    StreamCompleteData *data;

    BlockBackend *blk = s->common.blk;

    BlockDriverState *bs = blk_bs(blk);

    BlockDriverState *base = s->base;

    int64_t sector_num = 0;

    int64_t end = -1;

    uint64_t delay_ns = 0;

    int error = 0;

    int ret = 0;

    int n = 0;

    void *buf;



    if (!bs->backing) {

        goto out;

    }



    s->common.len = bdrv_getlength(bs);

    if (s->common.len < 0) {

        ret = s->common.len;

        goto out;

    }



    end = s->common.len >> BDRV_SECTOR_BITS;

    buf = qemu_blockalign(bs, STREAM_BUFFER_SIZE);



    /* Turn on copy-on-read for the whole block device so that guest read

     * requests help us make progress.  Only do this when copying the entire

     * backing chain since the copy-on-read operation does not take base into

     * account.

     */

    if (!base) {

        bdrv_enable_copy_on_read(bs);

    }



    for (sector_num = 0; sector_num < end; sector_num += n) {

        bool copy;



        /* Note that even when no rate limit is applied we need to yield

         * with no pending I/O here so that bdrv_drain_all() returns.

         */

        block_job_sleep_ns(&s->common, QEMU_CLOCK_REALTIME, delay_ns);

        if (block_job_is_cancelled(&s->common)) {

            break;

        }



        copy = false;



        ret = bdrv_is_allocated(bs, sector_num,

                                STREAM_BUFFER_SIZE / BDRV_SECTOR_SIZE, &n);

        if (ret == 1) {

            /* Allocated in the top, no need to copy.  */

        } else if (ret >= 0) {

            /* Copy if allocated in the intermediate images.  Limit to the

             * known-unallocated area [sector_num, sector_num+n).  */

            ret = bdrv_is_allocated_above(backing_bs(bs), base,

                                          sector_num, n, &n);



            /* Finish early if end of backing file has been reached */

            if (ret == 0 && n == 0) {

                n = end - sector_num;

            }



            copy = (ret == 1);

        }

        trace_stream_one_iteration(s, sector_num * BDRV_SECTOR_SIZE,

                                   n * BDRV_SECTOR_SIZE, ret);

        if (copy) {

            ret = stream_populate(blk, sector_num * BDRV_SECTOR_SIZE,

                                  n * BDRV_SECTOR_SIZE, buf);

        }

        if (ret < 0) {

            BlockErrorAction action =

                block_job_error_action(&s->common, s->on_error, true, -ret);

            if (action == BLOCK_ERROR_ACTION_STOP) {

                n = 0;

                continue;

            }

            if (error == 0) {

                error = ret;

            }

            if (action == BLOCK_ERROR_ACTION_REPORT) {

                break;

            }

        }

        ret = 0;



        /* Publish progress */

        s->common.offset += n * BDRV_SECTOR_SIZE;

        if (copy && s->common.speed) {

            delay_ns = ratelimit_calculate_delay(&s->limit,

                                                 n * BDRV_SECTOR_SIZE);

        }

    }



    if (!base) {

        bdrv_disable_copy_on_read(bs);

    }



    /* Do not remove the backing file if an error was there but ignored.  */

    ret = error;



    qemu_vfree(buf);



out:

    /* Modify backing chain and close BDSes in main loop */

    data = g_malloc(sizeof(*data));

    data->ret = ret;

    block_job_defer_to_main_loop(&s->common, stream_complete, data);

}
