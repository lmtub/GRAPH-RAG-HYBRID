static int vmdk_read(BlockDriverState *bs, int64_t sector_num, 

                    uint8_t *buf, int nb_sectors)

{

    BDRVVmdkState *s = bs->opaque;

    int index_in_cluster, n, ret;

    uint64_t cluster_offset;



    while (nb_sectors > 0) {

        cluster_offset = get_cluster_offset(bs, sector_num << 9, 0);

        index_in_cluster = sector_num % s->cluster_sectors;

        n = s->cluster_sectors - index_in_cluster;

        if (n > nb_sectors)

            n = nb_sectors;

        if (!cluster_offset) {

            // try to read from parent image, if exist

            if (s->hd->backing_hd) {

                if (!vmdk_is_cid_valid(bs))

                    return -1;

                ret = bdrv_read(s->hd->backing_hd, sector_num, buf, n);

                if (ret < 0)

                    return -1;

            } else {

                memset(buf, 0, 512 * n);

            }

        } else {

            if(bdrv_pread(s->hd, cluster_offset + index_in_cluster * 512, buf, n * 512) != n * 512)

                return -1;

        }

        nb_sectors -= n;

        sector_num += n;

        buf += n * 512;

    }

    return 0;

}
