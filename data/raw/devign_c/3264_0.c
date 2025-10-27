void qmp_block_set_io_throttle(const char *device, int64_t bps, int64_t bps_rd,

                               int64_t bps_wr,

                               int64_t iops,

                               int64_t iops_rd,

                               int64_t iops_wr,

                               bool has_bps_max,

                               int64_t bps_max,

                               bool has_bps_rd_max,

                               int64_t bps_rd_max,

                               bool has_bps_wr_max,

                               int64_t bps_wr_max,

                               bool has_iops_max,

                               int64_t iops_max,

                               bool has_iops_rd_max,

                               int64_t iops_rd_max,

                               bool has_iops_wr_max,

                               int64_t iops_wr_max,

                               bool has_iops_size,

                               int64_t iops_size,

                               bool has_group,

                               const char *group, Error **errp)

{

    ThrottleConfig cfg;

    BlockDriverState *bs;

    BlockBackend *blk;

    AioContext *aio_context;



    blk = blk_by_name(device);

    if (!blk) {

        error_set(errp, ERROR_CLASS_DEVICE_NOT_FOUND,

                  "Device '%s' not found", device);

        return;

    }



    aio_context = blk_get_aio_context(blk);

    aio_context_acquire(aio_context);



    bs = blk_bs(blk);

    if (!bs) {

        error_setg(errp, "Device '%s' has no medium", device);

        goto out;

    }



    memset(&cfg, 0, sizeof(cfg));

    cfg.buckets[THROTTLE_BPS_TOTAL].avg = bps;

    cfg.buckets[THROTTLE_BPS_READ].avg  = bps_rd;

    cfg.buckets[THROTTLE_BPS_WRITE].avg = bps_wr;



    cfg.buckets[THROTTLE_OPS_TOTAL].avg = iops;

    cfg.buckets[THROTTLE_OPS_READ].avg  = iops_rd;

    cfg.buckets[THROTTLE_OPS_WRITE].avg = iops_wr;



    if (has_bps_max) {

        cfg.buckets[THROTTLE_BPS_TOTAL].max = bps_max;

    }

    if (has_bps_rd_max) {

        cfg.buckets[THROTTLE_BPS_READ].max = bps_rd_max;

    }

    if (has_bps_wr_max) {

        cfg.buckets[THROTTLE_BPS_WRITE].max = bps_wr_max;

    }

    if (has_iops_max) {

        cfg.buckets[THROTTLE_OPS_TOTAL].max = iops_max;

    }

    if (has_iops_rd_max) {

        cfg.buckets[THROTTLE_OPS_READ].max = iops_rd_max;

    }

    if (has_iops_wr_max) {

        cfg.buckets[THROTTLE_OPS_WRITE].max = iops_wr_max;

    }



    if (has_iops_size) {

        cfg.op_size = iops_size;

    }



    if (!check_throttle_config(&cfg, errp)) {

        goto out;

    }



    if (throttle_enabled(&cfg)) {

        /* Enable I/O limits if they're not enabled yet, otherwise

         * just update the throttling group. */

        if (!bs->io_limits_enabled) {

            bdrv_io_limits_enable(bs, has_group ? group : device);

        } else if (has_group) {

            bdrv_io_limits_update_group(bs, group);

        }

        /* Set the new throttling configuration */

        bdrv_set_io_limits(bs, &cfg);

    } else if (bs->io_limits_enabled) {

        /* If all throttling settings are set to 0, disable I/O limits */

        bdrv_io_limits_disable(bs);

    }



out:

    aio_context_release(aio_context);

}
