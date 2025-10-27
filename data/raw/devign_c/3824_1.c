static int cow_create(const char *filename, QemuOpts *opts, Error **errp)

{

    struct cow_header_v2 cow_header;

    struct stat st;

    int64_t image_sectors = 0;

    char *image_filename = NULL;

    Error *local_err = NULL;

    int ret;

    BlockDriverState *cow_bs;



    /* Read out options */

    image_sectors = qemu_opt_get_size_del(opts, BLOCK_OPT_SIZE, 0) / 512;

    image_filename = qemu_opt_get_del(opts, BLOCK_OPT_BACKING_FILE);



    ret = bdrv_create_file(filename, opts, &local_err);

    if (ret < 0) {

        error_propagate(errp, local_err);

        goto exit;

    }



    cow_bs = NULL;

    ret = bdrv_open(&cow_bs, filename, NULL, NULL,

                    BDRV_O_RDWR | BDRV_O_PROTOCOL, NULL, &local_err);

    if (ret < 0) {

        error_propagate(errp, local_err);

        goto exit;

    }



    memset(&cow_header, 0, sizeof(cow_header));

    cow_header.magic = cpu_to_be32(COW_MAGIC);

    cow_header.version = cpu_to_be32(COW_VERSION);

    if (image_filename) {

        /* Note: if no file, we put a dummy mtime */

        cow_header.mtime = cpu_to_be32(0);



        if (stat(image_filename, &st) != 0) {

            goto mtime_fail;

        }

        cow_header.mtime = cpu_to_be32(st.st_mtime);

    mtime_fail:

        pstrcpy(cow_header.backing_file, sizeof(cow_header.backing_file),

                image_filename);

    }

    cow_header.sectorsize = cpu_to_be32(512);

    cow_header.size = cpu_to_be64(image_sectors * 512);

    ret = bdrv_pwrite(cow_bs, 0, &cow_header, sizeof(cow_header));

    if (ret < 0) {

        goto exit;

    }



    /* resize to include at least all the bitmap */

    ret = bdrv_truncate(cow_bs,

        sizeof(cow_header) + ((image_sectors + 7) >> 3));

    if (ret < 0) {

        goto exit;

    }



exit:

    g_free(image_filename);

    bdrv_unref(cow_bs);

    return ret;

}
