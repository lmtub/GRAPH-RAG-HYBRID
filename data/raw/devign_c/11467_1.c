static int vmdk_write(BlockDriverState *bs, int64_t sector_num,

                     const uint8_t *buf, int nb_sectors)

{

    BDRVVmdkState *s = bs->opaque;

    VmdkExtent *extent = NULL;

    int n, ret;

    int64_t index_in_cluster;

    uint64_t cluster_offset;

    VmdkMetaData m_data;



    if (sector_num > bs->total_sectors) {

        fprintf(stderr,

                "(VMDK) Wrong offset: sector_num=0x%" PRIx64

                " total_sectors=0x%" PRIx64 "\n",

                sector_num, bs->total_sectors);

        return -EIO;

    }



    while (nb_sectors > 0) {

        extent = find_extent(s, sector_num, extent);

        if (!extent) {

            return -EIO;

        }

        ret = get_cluster_offset(

                                bs,

                                extent,

                                &m_data,

                                sector_num << 9, !extent->compressed,

                                &cluster_offset);

        if (extent->compressed) {

            if (ret == 0) {

                /* Refuse write to allocated cluster for streamOptimized */

                fprintf(stderr,

                        "VMDK: can't write to allocated cluster"

                        " for streamOptimized\n");

                return -EIO;

            } else {

                /* allocate */

                ret = get_cluster_offset(

                                        bs,

                                        extent,

                                        &m_data,

                                        sector_num << 9, 1,

                                        &cluster_offset);

            }

        }

        if (ret) {

            return -EINVAL;

        }

        index_in_cluster = sector_num % extent->cluster_sectors;

        n = extent->cluster_sectors - index_in_cluster;

        if (n > nb_sectors) {

            n = nb_sectors;

        }



        ret = vmdk_write_extent(extent,

                        cluster_offset, index_in_cluster * 512,

                        buf, n, sector_num);

        if (ret) {

            return ret;

        }

        if (m_data.valid) {

            /* update L2 tables */

            if (vmdk_L2update(extent, &m_data) == -1) {

                return -EIO;

            }

        }

        nb_sectors -= n;

        sector_num += n;

        buf += n * 512;



        /* update CID on the first write every time the virtual disk is

         * opened */

        if (!s->cid_updated) {

            ret = vmdk_write_cid(bs, time(NULL));

            if (ret < 0) {

                return ret;

            }

            s->cid_updated = true;

        }

    }

    return 0;

}
