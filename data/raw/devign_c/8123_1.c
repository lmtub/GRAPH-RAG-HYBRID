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
                               int64_t iops_size, Error **errp)
{
    ThrottleConfig cfg;
    BlockDriverState *bs;
    AioContext *aio_context;
    bs = bdrv_find(device);
    if (!bs) {
        error_set(errp, QERR_DEVICE_NOT_FOUND, device);
        return;
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
        return;
    }
    aio_context = bdrv_get_aio_context(bs);
    aio_context_acquire(aio_context);
    if (!bs->io_limits_enabled && throttle_enabled(&cfg)) {
        bdrv_io_limits_enable(bs);
    } else if (bs->io_limits_enabled && !throttle_enabled(&cfg)) {
        bdrv_io_limits_disable(bs);
    }
    if (bs->io_limits_enabled) {
        bdrv_set_io_limits(bs, &cfg);
    }
}