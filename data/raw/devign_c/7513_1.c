void bdrv_query_info(BlockDriverState *bs,
                     BlockInfo **p_info,
                     Error **errp)
{
    BlockInfo *info = g_malloc0(sizeof(*info));
    BlockDriverState *bs0;
    ImageInfo **p_image_info;
    Error *local_err = NULL;
    info->device = g_strdup(bs->device_name);
    info->type = g_strdup("unknown");
    info->locked = bdrv_dev_is_medium_locked(bs);
    info->removable = bdrv_dev_has_removable_media(bs);
    if (bdrv_dev_has_removable_media(bs)) {
        info->has_tray_open = true;
        info->tray_open = bdrv_dev_is_tray_open(bs);
    }
    if (bdrv_iostatus_is_enabled(bs)) {
        info->has_io_status = true;
        info->io_status = bs->iostatus;
    }
    if (bs->dirty_bitmap) {
        info->has_dirty = true;
        info->dirty = g_malloc0(sizeof(*info->dirty));
        info->dirty->count = bdrv_get_dirty_count(bs) * BDRV_SECTOR_SIZE;
        info->dirty->granularity =
         ((int64_t) BDRV_SECTOR_SIZE << hbitmap_granularity(bs->dirty_bitmap));
    }
    if (bs->drv) {
        info->has_inserted = true;
        info->inserted = g_malloc0(sizeof(*info->inserted));
        info->inserted->file = g_strdup(bs->filename);
        info->inserted->ro = bs->read_only;
        info->inserted->drv = g_strdup(bs->drv->format_name);
        info->inserted->encrypted = bs->encrypted;
        info->inserted->encryption_key_missing = bdrv_key_required(bs);
        if (bs->backing_file[0]) {
            info->inserted->has_backing_file = true;
            info->inserted->backing_file = g_strdup(bs->backing_file);
        }
        info->inserted->backing_file_depth = bdrv_get_backing_file_depth(bs);
        if (bs->io_limits_enabled) {
            ThrottleConfig cfg;
            throttle_get_config(&bs->throttle_state, &cfg);
            info->inserted->bps     = cfg.buckets[THROTTLE_BPS_TOTAL].avg;
            info->inserted->bps_rd  = cfg.buckets[THROTTLE_BPS_READ].avg;
            info->inserted->bps_wr  = cfg.buckets[THROTTLE_BPS_WRITE].avg;
            info->inserted->iops    = cfg.buckets[THROTTLE_OPS_TOTAL].avg;
            info->inserted->iops_rd = cfg.buckets[THROTTLE_OPS_READ].avg;
            info->inserted->iops_wr = cfg.buckets[THROTTLE_OPS_WRITE].avg;
        }
        bs0 = bs;
        p_image_info = &info->inserted->image;
        while (1) {
            bdrv_query_image_info(bs0, p_image_info, &local_err);
            if (error_is_set(&local_err)) {
                error_propagate(errp, local_err);
                goto err;
            }
            if (bs0->drv && bs0->backing_hd) {
                bs0 = bs0->backing_hd;
                (*p_image_info)->has_backing_image = true;
                p_image_info = &((*p_image_info)->backing_image);
            } else {
                break;
            }
        }
    }
    *p_info = info;
    return;
 err:
    qapi_free_BlockInfo(info);
}