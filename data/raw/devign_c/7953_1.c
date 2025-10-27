static int qcow2_update_ext_header(BlockDriverState *bs,

    const char *backing_file, const char *backing_fmt)

{

    size_t backing_file_len = 0;

    size_t backing_fmt_len = 0;

    BDRVQcowState *s = bs->opaque;

    QCowExtension ext_backing_fmt = {0, 0};

    int ret;



    /* Backing file format doesn't make sense without a backing file */

    if (backing_fmt && !backing_file) {

        return -EINVAL;

    }



    /* Prepare the backing file format extension if needed */

    if (backing_fmt) {

        ext_backing_fmt.len = cpu_to_be32(strlen(backing_fmt));

        ext_backing_fmt.magic = cpu_to_be32(QCOW_EXT_MAGIC_BACKING_FORMAT);

        backing_fmt_len = ((sizeof(ext_backing_fmt)

            + strlen(backing_fmt) + 7) & ~7);

    }



    /* Check if we can fit the new header into the first cluster */

    if (backing_file) {

        backing_file_len = strlen(backing_file);

    }



    size_t header_size = sizeof(QCowHeader) + backing_file_len

        + backing_fmt_len;



    if (header_size > s->cluster_size) {

        return -ENOSPC;

    }



    /* Rewrite backing file name and qcow2 extensions */

    size_t ext_size = header_size - sizeof(QCowHeader);

    uint8_t buf[ext_size];

    size_t offset = 0;

    size_t backing_file_offset = 0;



    if (backing_file) {

        if (backing_fmt) {

            int padding = backing_fmt_len -

                (sizeof(ext_backing_fmt) + strlen(backing_fmt));



            memcpy(buf + offset, &ext_backing_fmt, sizeof(ext_backing_fmt));

            offset += sizeof(ext_backing_fmt);



            memcpy(buf + offset, backing_fmt, strlen(backing_fmt));

            offset += strlen(backing_fmt);



            memset(buf + offset, 0, padding);

            offset += padding;

        }



        memcpy(buf + offset, backing_file, backing_file_len);

        backing_file_offset = sizeof(QCowHeader) + offset;

    }



    ret = bdrv_pwrite(bs->file, sizeof(QCowHeader), buf, ext_size);

    if (ret < 0) {

        goto fail;

    }



    /* Update header fields */

    uint64_t be_backing_file_offset = cpu_to_be64(backing_file_offset);

    uint32_t be_backing_file_size = cpu_to_be32(backing_file_len);



    ret = bdrv_pwrite(bs->file, offsetof(QCowHeader, backing_file_offset),

        &be_backing_file_offset, sizeof(uint64_t));

    if (ret < 0) {

        goto fail;

    }



    ret = bdrv_pwrite(bs->file, offsetof(QCowHeader, backing_file_size),

        &be_backing_file_size, sizeof(uint32_t));

    if (ret < 0) {

        goto fail;

    }



    ret = 0;

fail:

    return ret;

}
