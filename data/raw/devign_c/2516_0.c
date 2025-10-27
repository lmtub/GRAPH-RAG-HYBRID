static int get_whole_cluster(BlockDriverState *bs,

                VmdkExtent *extent,

                uint64_t cluster_offset,

                uint64_t offset,

                bool allocate)

{

    /* 128 sectors * 512 bytes each = grain size 64KB */

    uint8_t  whole_grain[extent->cluster_sectors * 512];



    /* we will be here if it's first write on non-exist grain(cluster).

     * try to read from parent image, if exist */

    if (bs->backing_hd) {

        int ret;



        if (!vmdk_is_cid_valid(bs))

            return -1;



        /* floor offset to cluster */

        offset -= offset % (extent->cluster_sectors * 512);

        ret = bdrv_read(bs->backing_hd, offset >> 9, whole_grain,

                extent->cluster_sectors);

        if (ret < 0) {

            return -1;

        }



        /* Write grain only into the active image */

        ret = bdrv_write(extent->file, cluster_offset, whole_grain,

                extent->cluster_sectors);

        if (ret < 0) {

            return -1;

        }

    }

    return 0;

}
