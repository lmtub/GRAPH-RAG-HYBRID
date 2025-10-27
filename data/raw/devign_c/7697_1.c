static int vmdk_write_extent(VmdkExtent *extent, int64_t cluster_offset,

                            int64_t offset_in_cluster, const uint8_t *buf,

                            int nb_sectors, int64_t sector_num)

{

    int ret;

    VmdkGrainMarker *data = NULL;

    uLongf buf_len;

    const uint8_t *write_buf = buf;

    int write_len = nb_sectors * 512;



    if (extent->compressed) {

        if (!extent->has_marker) {

            ret = -EINVAL;

            goto out;

        }

        buf_len = (extent->cluster_sectors << 9) * 2;

        data = g_malloc(buf_len + sizeof(VmdkGrainMarker));

        if (compress(data->data, &buf_len, buf, nb_sectors << 9) != Z_OK ||

                buf_len == 0) {

            ret = -EINVAL;

            goto out;

        }

        data->lba = sector_num;

        data->size = buf_len;

        write_buf = (uint8_t *)data;

        write_len = buf_len + sizeof(VmdkGrainMarker);

    }

    ret = bdrv_pwrite(extent->file,

                        cluster_offset + offset_in_cluster,

                        write_buf,

                        write_len);

    if (ret != write_len) {

        ret = ret < 0 ? ret : -EIO;

        goto out;

    }

    ret = 0;

 out:

    g_free(data);

    return ret;

}
