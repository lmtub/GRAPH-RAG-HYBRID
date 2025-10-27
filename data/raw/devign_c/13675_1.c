BlockDeviceInfo *bdrv_block_device_info(BlockDriverState *bs)

{

    BlockDeviceInfo *info = g_malloc0(sizeof(*info));



    info->file                   = g_strdup(bs->filename);

    info->ro                     = bs->read_only;

    info->drv                    = g_strdup(bs->drv->format_name);

    info->encrypted              = bs->encrypted;

    info->encryption_key_missing = bdrv_key_required(bs);



    info->cache = g_new(BlockdevCacheInfo, 1);

    *info->cache = (BlockdevCacheInfo) {

        .writeback      = bdrv_enable_write_cache(bs),

        .direct         = !!(bs->open_flags & BDRV_O_NOCACHE),

        .no_flush       = !!(bs->open_flags & BDRV_O_NO_FLUSH),

    };



    if (bs->node_name[0]) {

        info->has_node_name = true;

        info->node_name = g_strdup(bs->node_name);

    }



    if (bs->backing_file[0]) {

        info->has_backing_file = true;

        info->backing_file = g_strdup(bs->backing_file);

    }



    info->backing_file_depth = bdrv_get_backing_file_depth(bs);

    info->detect_zeroes = bs->detect_zeroes;



    if (bs->io_limits_enabled) {

        ThrottleConfig cfg;

        throttle_get_config(&bs->throttle_state, &cfg);

        info->bps     = cfg.buckets[THROTTLE_BPS_TOTAL].avg;

        info->bps_rd  = cfg.buckets[THROTTLE_BPS_READ].avg;

        info->bps_wr  = cfg.buckets[THROTTLE_BPS_WRITE].avg;



        info->iops    = cfg.buckets[THROTTLE_OPS_TOTAL].avg;

        info->iops_rd = cfg.buckets[THROTTLE_OPS_READ].avg;

        info->iops_wr = cfg.buckets[THROTTLE_OPS_WRITE].avg;



        info->has_bps_max     = cfg.buckets[THROTTLE_BPS_TOTAL].max;

        info->bps_max         = cfg.buckets[THROTTLE_BPS_TOTAL].max;

        info->has_bps_rd_max  = cfg.buckets[THROTTLE_BPS_READ].max;

        info->bps_rd_max      = cfg.buckets[THROTTLE_BPS_READ].max;

        info->has_bps_wr_max  = cfg.buckets[THROTTLE_BPS_WRITE].max;

        info->bps_wr_max      = cfg.buckets[THROTTLE_BPS_WRITE].max;



        info->has_iops_max    = cfg.buckets[THROTTLE_OPS_TOTAL].max;

        info->iops_max        = cfg.buckets[THROTTLE_OPS_TOTAL].max;

        info->has_iops_rd_max = cfg.buckets[THROTTLE_OPS_READ].max;

        info->iops_rd_max     = cfg.buckets[THROTTLE_OPS_READ].max;

        info->has_iops_wr_max = cfg.buckets[THROTTLE_OPS_WRITE].max;

        info->iops_wr_max     = cfg.buckets[THROTTLE_OPS_WRITE].max;



        info->has_iops_size = cfg.op_size;

        info->iops_size = cfg.op_size;

    }



    info->write_threshold = bdrv_write_threshold_get(bs);



    return info;

}
