int coroutine_fn blk_co_preadv(BlockBackend *blk, int64_t offset,

                               unsigned int bytes, QEMUIOVector *qiov,

                               BdrvRequestFlags flags)

{

    int ret;



    trace_blk_co_preadv(blk, blk_bs(blk), offset, bytes, flags);



    ret = blk_check_byte_request(blk, offset, bytes);

    if (ret < 0) {

        return ret;

    }



    /* throttling disk I/O */

    if (blk->public.throttle_state) {

        throttle_group_co_io_limits_intercept(blk, bytes, false);

    }



    return bdrv_co_preadv(blk_bs(blk), offset, bytes, qiov, flags);

}
