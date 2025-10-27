static bool bdrv_exceed_io_limits(BlockDriverState *bs, int nb_sectors,

                           bool is_write, int64_t *wait)

{

    int64_t  now, max_wait;

    uint64_t bps_wait = 0, iops_wait = 0;

    double   elapsed_time;

    int      bps_ret, iops_ret;



    now = qemu_get_clock_ns(vm_clock);

    if ((bs->slice_start < now)

        && (bs->slice_end > now)) {

        bs->slice_end = now + BLOCK_IO_SLICE_TIME;

    } else {

        bs->slice_start = now;

        bs->slice_end   = now + BLOCK_IO_SLICE_TIME;

        memset(&bs->slice_submitted, 0, sizeof(bs->slice_submitted));

    }



    elapsed_time  = now - bs->slice_start;

    elapsed_time  /= (NANOSECONDS_PER_SECOND);



    bps_ret  = bdrv_exceed_bps_limits(bs, nb_sectors,

                                      is_write, elapsed_time, &bps_wait);

    iops_ret = bdrv_exceed_iops_limits(bs, is_write,

                                      elapsed_time, &iops_wait);

    if (bps_ret || iops_ret) {

        max_wait = bps_wait > iops_wait ? bps_wait : iops_wait;

        if (wait) {

            *wait = max_wait;

        }



        now = qemu_get_clock_ns(vm_clock);

        if (bs->slice_end < now + max_wait) {

            bs->slice_end = now + max_wait;

        }



        return true;

    }



    if (wait) {

        *wait = 0;

    }



    bs->slice_submitted.bytes[is_write] += (int64_t)nb_sectors *

                                           BDRV_SECTOR_SIZE;

    bs->slice_submitted.ios[is_write]++;



    return false;

}
