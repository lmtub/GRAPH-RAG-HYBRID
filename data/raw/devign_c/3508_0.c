void bdrv_query_image_info(BlockDriverState *bs,

                           ImageInfo **p_info,

                           Error **errp)

{

    int64_t size;

    const char *backing_filename;

    BlockDriverInfo bdi;

    int ret;

    Error *err = NULL;

    ImageInfo *info;



    size = bdrv_getlength(bs);

    if (size < 0) {

        error_setg_errno(errp, -size, "Can't get size of device '%s'",

                         bdrv_get_device_name(bs));

        return;

    }



    info = g_new0(ImageInfo, 1);

    info->filename        = g_strdup(bs->filename);

    info->format          = g_strdup(bdrv_get_format_name(bs));

    info->virtual_size    = size;

    info->actual_size     = bdrv_get_allocated_file_size(bs);

    info->has_actual_size = info->actual_size >= 0;

    if (bdrv_is_encrypted(bs)) {

        info->encrypted = true;

        info->has_encrypted = true;

    }

    if (bdrv_get_info(bs, &bdi) >= 0) {

        if (bdi.cluster_size != 0) {

            info->cluster_size = bdi.cluster_size;

            info->has_cluster_size = true;

        }

        info->dirty_flag = bdi.is_dirty;

        info->has_dirty_flag = true;

    }

    info->format_specific     = bdrv_get_specific_info(bs);

    info->has_format_specific = info->format_specific != NULL;



    backing_filename = bs->backing_file;

    if (backing_filename[0] != '\0') {

        char *backing_filename2 = g_malloc0(1024);

        info->backing_filename = g_strdup(backing_filename);

        info->has_backing_filename = true;

        bdrv_get_full_backing_filename(bs, backing_filename2, 1024, &err);

        if (err) {

            error_propagate(errp, err);

            qapi_free_ImageInfo(info);

            g_free(backing_filename2);

            return;

        }



        if (strcmp(backing_filename, backing_filename2) != 0) {

            info->full_backing_filename =

                        g_strdup(backing_filename2);

            info->has_full_backing_filename = true;

        }



        if (bs->backing_format[0]) {

            info->backing_filename_format = g_strdup(bs->backing_format);

            info->has_backing_filename_format = true;

        }

        g_free(backing_filename2);

    }



    ret = bdrv_query_snapshot_info_list(bs, &info->snapshots, &err);

    switch (ret) {

    case 0:

        if (info->snapshots) {

            info->has_snapshots = true;

        }

        break;

    /* recoverable error */

    case -ENOMEDIUM:

    case -ENOTSUP:

        error_free(err);

        break;

    default:

        error_propagate(errp, err);

        qapi_free_ImageInfo(info);

        return;

    }



    *p_info = info;

}
