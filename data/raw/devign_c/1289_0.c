static int vmdk_read_extent(VmdkExtent *extent, int64_t cluster_offset,

                            int64_t offset_in_cluster, QEMUIOVector *qiov,

                            int bytes)

{

    int ret;

    int cluster_bytes, buf_bytes;

    uint8_t *cluster_buf, *compressed_data;

    uint8_t *uncomp_buf;

    uint32_t data_len;

    VmdkGrainMarker *marker;

    uLongf buf_len;





    if (!extent->compressed) {

        ret = bdrv_co_preadv(extent->file->bs,

                             cluster_offset + offset_in_cluster, bytes,

                             qiov, 0);

        if (ret < 0) {

            return ret;

        }

        return 0;

    }

    cluster_bytes = extent->cluster_sectors * 512;

    /* Read two clusters in case GrainMarker + compressed data > one cluster */

    buf_bytes = cluster_bytes * 2;

    cluster_buf = g_malloc(buf_bytes);

    uncomp_buf = g_malloc(cluster_bytes);

    ret = bdrv_pread(extent->file,

                cluster_offset,

                cluster_buf, buf_bytes);

    if (ret < 0) {

        goto out;

    }

    compressed_data = cluster_buf;

    buf_len = cluster_bytes;

    data_len = cluster_bytes;

    if (extent->has_marker) {

        marker = (VmdkGrainMarker *)cluster_buf;

        compressed_data = marker->data;

        data_len = le32_to_cpu(marker->size);

    }

    if (!data_len || data_len > buf_bytes) {

        ret = -EINVAL;

        goto out;

    }

    ret = uncompress(uncomp_buf, &buf_len, compressed_data, data_len);

    if (ret != Z_OK) {

        ret = -EINVAL;

        goto out;



    }

    if (offset_in_cluster < 0 ||

            offset_in_cluster + bytes > buf_len) {

        ret = -EINVAL;

        goto out;

    }

    qemu_iovec_from_buf(qiov, 0, uncomp_buf + offset_in_cluster, bytes);

    ret = 0;



 out:

    g_free(uncomp_buf);

    g_free(cluster_buf);

    return ret;

}
