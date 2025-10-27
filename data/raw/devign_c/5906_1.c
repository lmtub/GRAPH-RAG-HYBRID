static int vdi_create(const char *filename, QemuOpts *opts, Error **errp)

{

    int ret = 0;

    uint64_t bytes = 0;

    uint32_t blocks;

    size_t block_size = DEFAULT_CLUSTER_SIZE;

    uint32_t image_type = VDI_TYPE_DYNAMIC;

    VdiHeader header;

    size_t i;

    size_t bmap_size;

    int64_t offset = 0;

    Error *local_err = NULL;

    BlockDriverState *bs = NULL;

    uint32_t *bmap = NULL;



    logout("\n");



    /* Read out options. */

    bytes = qemu_opt_get_size_del(opts, BLOCK_OPT_SIZE, 0);

#if defined(CONFIG_VDI_BLOCK_SIZE)

    /* TODO: Additional checks (SECTOR_SIZE * 2^n, ...). */

    block_size = qemu_opt_get_size_del(opts,

                                       BLOCK_OPT_CLUSTER_SIZE,

                                       DEFAULT_CLUSTER_SIZE);

#endif

#if defined(CONFIG_VDI_STATIC_IMAGE)

    if (qemu_opt_get_bool_del(opts, BLOCK_OPT_STATIC, false)) {

        image_type = VDI_TYPE_STATIC;

    }

#endif



    if (bytes > VDI_DISK_SIZE_MAX) {

        ret = -ENOTSUP;

        error_setg(errp, "Unsupported VDI image size (size is 0x%" PRIx64

                          ", max supported is 0x%" PRIx64 ")",

                          bytes, VDI_DISK_SIZE_MAX);

        goto exit;

    }



    ret = bdrv_create_file(filename, opts, &local_err);

    if (ret < 0) {

        error_propagate(errp, local_err);

        goto exit;

    }

    ret = bdrv_open(&bs, filename, NULL, NULL, BDRV_O_RDWR | BDRV_O_PROTOCOL,

                    NULL, &local_err);

    if (ret < 0) {

        error_propagate(errp, local_err);

        goto exit;

    }



    /* We need enough blocks to store the given disk size,

       so always round up. */

    blocks = (bytes + block_size - 1) / block_size;



    bmap_size = blocks * sizeof(uint32_t);

    bmap_size = ((bmap_size + SECTOR_SIZE - 1) & ~(SECTOR_SIZE -1));



    memset(&header, 0, sizeof(header));

    pstrcpy(header.text, sizeof(header.text), VDI_TEXT);

    header.signature = VDI_SIGNATURE;

    header.version = VDI_VERSION_1_1;

    header.header_size = 0x180;

    header.image_type = image_type;

    header.offset_bmap = 0x200;

    header.offset_data = 0x200 + bmap_size;

    header.sector_size = SECTOR_SIZE;

    header.disk_size = bytes;

    header.block_size = block_size;

    header.blocks_in_image = blocks;

    if (image_type == VDI_TYPE_STATIC) {

        header.blocks_allocated = blocks;

    }

    uuid_generate(header.uuid_image);

    uuid_generate(header.uuid_last_snap);

    /* There is no need to set header.uuid_link or header.uuid_parent here. */

#if defined(CONFIG_VDI_DEBUG)

    vdi_header_print(&header);

#endif

    vdi_header_to_le(&header);

    ret = bdrv_pwrite_sync(bs, offset, &header, sizeof(header));

    if (ret < 0) {

        error_setg(errp, "Error writing header to %s", filename);

        goto exit;

    }

    offset += sizeof(header);



    if (bmap_size > 0) {

        bmap = g_malloc0(bmap_size);

        for (i = 0; i < blocks; i++) {

            if (image_type == VDI_TYPE_STATIC) {

                bmap[i] = i;

            } else {

                bmap[i] = VDI_UNALLOCATED;

            }

        }

        ret = bdrv_pwrite_sync(bs, offset, bmap, bmap_size);

        if (ret < 0) {

            error_setg(errp, "Error writing bmap to %s", filename);

            goto exit;

        }

        offset += bmap_size;

    }



    if (image_type == VDI_TYPE_STATIC) {

        ret = bdrv_truncate(bs, offset + blocks * block_size);

        if (ret < 0) {

            error_setg(errp, "Failed to statically allocate %s", filename);

            goto exit;

        }

    }



exit:

    bdrv_unref(bs);

    g_free(bmap);

    return ret;

}
