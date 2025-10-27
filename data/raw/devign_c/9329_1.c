static int vmdk_create_extent(const char *filename, int64_t filesize,

                              bool flat, bool compress, bool zeroed_grain,

                              Error **errp)

{

    int ret, i;

    BlockDriverState *bs = NULL;

    VMDK4Header header;

    Error *local_err;

    uint32_t tmp, magic, grains, gd_sectors, gt_size, gt_count;

    uint32_t *gd_buf = NULL;

    int gd_buf_size;



    ret = bdrv_create_file(filename, NULL, &local_err);

    if (ret < 0) {

        error_propagate(errp, local_err);

        goto exit;

    }



    ret = bdrv_file_open(&bs, filename, NULL, NULL, BDRV_O_RDWR, &local_err);

    if (ret < 0) {

        error_propagate(errp, local_err);

        goto exit;

    }



    if (flat) {

        ret = bdrv_truncate(bs, filesize);

        if (ret < 0) {

            error_setg(errp, "Could not truncate file");

        }

        goto exit;

    }

    magic = cpu_to_be32(VMDK4_MAGIC);

    memset(&header, 0, sizeof(header));

    header.version = zeroed_grain ? 2 : 1;

    header.flags = VMDK4_FLAG_RGD | VMDK4_FLAG_NL_DETECT

                   | (compress ? VMDK4_FLAG_COMPRESS | VMDK4_FLAG_MARKER : 0)

                   | (zeroed_grain ? VMDK4_FLAG_ZERO_GRAIN : 0);

    header.compressAlgorithm = compress ? VMDK4_COMPRESSION_DEFLATE : 0;

    header.capacity = filesize / BDRV_SECTOR_SIZE;

    header.granularity = 128;

    header.num_gtes_per_gt = BDRV_SECTOR_SIZE;



    grains = DIV_ROUND_UP(filesize / BDRV_SECTOR_SIZE, header.granularity);

    gt_size = DIV_ROUND_UP(header.num_gtes_per_gt * sizeof(uint32_t),

                           BDRV_SECTOR_SIZE);

    gt_count = DIV_ROUND_UP(grains, header.num_gtes_per_gt);

    gd_sectors = DIV_ROUND_UP(gt_count * sizeof(uint32_t), BDRV_SECTOR_SIZE);



    header.desc_offset = 1;

    header.desc_size = 20;

    header.rgd_offset = header.desc_offset + header.desc_size;

    header.gd_offset = header.rgd_offset + gd_sectors + (gt_size * gt_count);

    header.grain_offset =

        ROUND_UP(header.gd_offset + gd_sectors + (gt_size * gt_count),

                 header.granularity);

    /* swap endianness for all header fields */

    header.version = cpu_to_le32(header.version);

    header.flags = cpu_to_le32(header.flags);

    header.capacity = cpu_to_le64(header.capacity);

    header.granularity = cpu_to_le64(header.granularity);

    header.num_gtes_per_gt = cpu_to_le32(header.num_gtes_per_gt);

    header.desc_offset = cpu_to_le64(header.desc_offset);

    header.desc_size = cpu_to_le64(header.desc_size);

    header.rgd_offset = cpu_to_le64(header.rgd_offset);

    header.gd_offset = cpu_to_le64(header.gd_offset);

    header.grain_offset = cpu_to_le64(header.grain_offset);

    header.compressAlgorithm = cpu_to_le16(header.compressAlgorithm);



    header.check_bytes[0] = 0xa;

    header.check_bytes[1] = 0x20;

    header.check_bytes[2] = 0xd;

    header.check_bytes[3] = 0xa;



    /* write all the data */

    ret = bdrv_pwrite(bs, 0, &magic, sizeof(magic));

    if (ret < 0) {

        error_set(errp, QERR_IO_ERROR);

        goto exit;

    }

    ret = bdrv_pwrite(bs, sizeof(magic), &header, sizeof(header));

    if (ret < 0) {

        error_set(errp, QERR_IO_ERROR);

        goto exit;

    }



    ret = bdrv_truncate(bs, le64_to_cpu(header.grain_offset) << 9);

    if (ret < 0) {

        error_setg(errp, "Could not truncate file");

        goto exit;

    }



    /* write grain directory */

    gd_buf_size = gd_sectors * BDRV_SECTOR_SIZE;

    gd_buf = g_malloc0(gd_buf_size);

    for (i = 0, tmp = le64_to_cpu(header.rgd_offset) + gd_sectors;

         i < gt_count; i++, tmp += gt_size) {

        gd_buf[i] = cpu_to_le32(tmp);

    }

    ret = bdrv_pwrite(bs, le64_to_cpu(header.rgd_offset) * BDRV_SECTOR_SIZE,

                      gd_buf, gd_buf_size);

    if (ret < 0) {

        error_set(errp, QERR_IO_ERROR);

        goto exit;

    }



    /* write backup grain directory */

    for (i = 0, tmp = le64_to_cpu(header.gd_offset) + gd_sectors;

         i < gt_count; i++, tmp += gt_size) {

        gd_buf[i] = cpu_to_le32(tmp);

    }

    ret = bdrv_pwrite(bs, le64_to_cpu(header.gd_offset) * BDRV_SECTOR_SIZE,

                      gd_buf, gd_buf_size);

    if (ret < 0) {

        error_set(errp, QERR_IO_ERROR);

        goto exit;

    }



    ret = 0;

exit:

    if (bs) {

        bdrv_unref(bs);

    }

    g_free(gd_buf);

    return ret;

}
