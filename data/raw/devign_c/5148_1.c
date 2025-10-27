qcow_co_pwritev_compressed(BlockDriverState *bs, uint64_t offset,

                           uint64_t bytes, QEMUIOVector *qiov)

{

    BDRVQcowState *s = bs->opaque;

    QEMUIOVector hd_qiov;

    struct iovec iov;

    z_stream strm;

    int ret, out_len;

    uint8_t *buf, *out_buf;

    uint64_t cluster_offset;



    buf = qemu_blockalign(bs, s->cluster_size);

    if (bytes != s->cluster_size) {

        if (bytes > s->cluster_size ||

            offset + bytes != bs->total_sectors << BDRV_SECTOR_BITS)

        {

            qemu_vfree(buf);

            return -EINVAL;

        }

        /* Zero-pad last write if image size is not cluster aligned */

        memset(buf + bytes, 0, s->cluster_size - bytes);

    }

    qemu_iovec_to_buf(qiov, 0, buf, qiov->size);



    out_buf = g_malloc(s->cluster_size);



    /* best compression, small window, no zlib header */

    memset(&strm, 0, sizeof(strm));

    ret = deflateInit2(&strm, Z_DEFAULT_COMPRESSION,

                       Z_DEFLATED, -12,

                       9, Z_DEFAULT_STRATEGY);

    if (ret != 0) {

        ret = -EINVAL;

        goto fail;

    }



    strm.avail_in = s->cluster_size;

    strm.next_in = (uint8_t *)buf;

    strm.avail_out = s->cluster_size;

    strm.next_out = out_buf;



    ret = deflate(&strm, Z_FINISH);

    if (ret != Z_STREAM_END && ret != Z_OK) {

        deflateEnd(&strm);

        ret = -EINVAL;

        goto fail;

    }

    out_len = strm.next_out - out_buf;



    deflateEnd(&strm);



    if (ret != Z_STREAM_END || out_len >= s->cluster_size) {

        /* could not compress: write normal cluster */

        ret = qcow_co_writev(bs, offset >> BDRV_SECTOR_BITS,

                             bytes >> BDRV_SECTOR_BITS, qiov);

        if (ret < 0) {

            goto fail;

        }

        goto success;

    }

    qemu_co_mutex_lock(&s->lock);

    cluster_offset = get_cluster_offset(bs, offset, 2, out_len, 0, 0);

    qemu_co_mutex_unlock(&s->lock);

    if (cluster_offset == 0) {

        ret = -EIO;

        goto fail;

    }

    cluster_offset &= s->cluster_offset_mask;



    iov = (struct iovec) {

        .iov_base   = out_buf,

        .iov_len    = out_len,

    };

    qemu_iovec_init_external(&hd_qiov, &iov, 1);

    ret = bdrv_co_pwritev(bs->file, cluster_offset, out_len, &hd_qiov, 0);

    if (ret < 0) {

        goto fail;

    }

success:

    ret = 0;

fail:

    qemu_vfree(buf);

    g_free(out_buf);

    return ret;

}
