static int vhdx_create(const char *filename, QemuOpts *opts, Error **errp)

{

    int ret = 0;

    uint64_t image_size = (uint64_t) 2 * GiB;

    uint32_t log_size   = 1 * MiB;

    uint32_t block_size = 0;

    uint64_t signature;

    uint64_t metadata_offset;

    bool use_zero_blocks = false;



    gunichar2 *creator = NULL;

    glong creator_items;

    BlockBackend *blk;

    char *type = NULL;

    VHDXImageType image_type;

    Error *local_err = NULL;



    image_size = ROUND_UP(qemu_opt_get_size_del(opts, BLOCK_OPT_SIZE, 0),

                          BDRV_SECTOR_SIZE);

    log_size = qemu_opt_get_size_del(opts, VHDX_BLOCK_OPT_LOG_SIZE, 0);

    block_size = qemu_opt_get_size_del(opts, VHDX_BLOCK_OPT_BLOCK_SIZE, 0);

    type = qemu_opt_get_del(opts, BLOCK_OPT_SUBFMT);

    use_zero_blocks = qemu_opt_get_bool_del(opts, VHDX_BLOCK_OPT_ZERO, true);



    if (image_size > VHDX_MAX_IMAGE_SIZE) {

        error_setg_errno(errp, EINVAL, "Image size too large; max of 64TB");

        ret = -EINVAL;

        goto exit;

    }



    if (type == NULL) {

        type = g_strdup("dynamic");

    }



    if (!strcmp(type, "dynamic")) {

        image_type = VHDX_TYPE_DYNAMIC;

    } else if (!strcmp(type, "fixed")) {

        image_type = VHDX_TYPE_FIXED;

    } else if (!strcmp(type, "differencing")) {

        error_setg_errno(errp, ENOTSUP,

                         "Differencing files not yet supported");

        ret = -ENOTSUP;

        goto exit;

    } else {

        ret = -EINVAL;

        goto exit;

    }



    /* These are pretty arbitrary, and mainly designed to keep the BAT

     * size reasonable to load into RAM */

    if (block_size == 0) {

        if (image_size > 32 * TiB) {

            block_size = 64 * MiB;

        } else if (image_size > (uint64_t) 100 * GiB) {

            block_size = 32 * MiB;

        } else if (image_size > 1 * GiB) {

            block_size = 16 * MiB;

        } else {

            block_size = 8 * MiB;

        }

    }





    /* make the log size close to what was specified, but must be

     * min 1MB, and multiple of 1MB */

    log_size = ROUND_UP(log_size, MiB);



    block_size = ROUND_UP(block_size, MiB);

    block_size = block_size > VHDX_BLOCK_SIZE_MAX ? VHDX_BLOCK_SIZE_MAX :

                                                    block_size;



    ret = bdrv_create_file(filename, opts, &local_err);

    if (ret < 0) {

        error_propagate(errp, local_err);

        goto exit;

    }



    blk = blk_new_open(filename, NULL, NULL,

                       BDRV_O_RDWR | BDRV_O_PROTOCOL, &local_err);

    if (blk == NULL) {

        error_propagate(errp, local_err);

        ret = -EIO;

        goto exit;

    }



    blk_set_allow_write_beyond_eof(blk, true);



    /* Create (A) */



    /* The creator field is optional, but may be useful for

     * debugging / diagnostics */

    creator = g_utf8_to_utf16("QEMU v" QEMU_VERSION, -1, NULL,

                              &creator_items, NULL);

    signature = cpu_to_le64(VHDX_FILE_SIGNATURE);

    ret = blk_pwrite(blk, VHDX_FILE_ID_OFFSET, &signature, sizeof(signature),

                     0);

    if (ret < 0) {

        goto delete_and_exit;

    }

    if (creator) {

        ret = blk_pwrite(blk, VHDX_FILE_ID_OFFSET + sizeof(signature),

                         creator, creator_items * sizeof(gunichar2), 0);

        if (ret < 0) {

            goto delete_and_exit;

        }

    }





    /* Creates (B),(C) */

    ret = vhdx_create_new_headers(blk_bs(blk), image_size, log_size);

    if (ret < 0) {

        goto delete_and_exit;

    }



    /* Creates (D),(E),(G) explicitly. (F) created as by-product */

    ret = vhdx_create_new_region_table(blk_bs(blk), image_size, block_size, 512,

                                       log_size, use_zero_blocks, image_type,

                                       &metadata_offset);

    if (ret < 0) {

        goto delete_and_exit;

    }



    /* Creates (H) */

    ret = vhdx_create_new_metadata(blk_bs(blk), image_size, block_size, 512,

                                   metadata_offset, image_type);

    if (ret < 0) {

        goto delete_and_exit;

    }





delete_and_exit:

    blk_unref(blk);

exit:

    g_free(type);

    g_free(creator);

    return ret;

}
