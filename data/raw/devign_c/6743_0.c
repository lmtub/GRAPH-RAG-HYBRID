static int vmdk_add_extent(BlockDriverState *bs,

                           BlockDriverState *file, bool flat, int64_t sectors,

                           int64_t l1_offset, int64_t l1_backup_offset,

                           uint32_t l1_size,

                           int l2_size, uint64_t cluster_sectors,

                           VmdkExtent **new_extent,

                           Error **errp)

{

    VmdkExtent *extent;

    BDRVVmdkState *s = bs->opaque;



    if (cluster_sectors > 0x200000) {

        /* 0x200000 * 512Bytes = 1GB for one cluster is unrealistic */

        error_setg(errp, "Invalid granularity, image may be corrupt");

        return -EFBIG;

    }

    if (l1_size > 512 * 1024 * 1024) {

        /* Although with big capacity and small l1_entry_sectors, we can get a

         * big l1_size, we don't want unbounded value to allocate the table.

         * Limit it to 512M, which is 16PB for default cluster and L2 table

         * size */

        error_setg(errp, "L1 size too big");

        return -EFBIG;

    }



    s->extents = g_realloc(s->extents,

                              (s->num_extents + 1) * sizeof(VmdkExtent));

    extent = &s->extents[s->num_extents];

    s->num_extents++;



    memset(extent, 0, sizeof(VmdkExtent));

    extent->file = file;

    extent->flat = flat;

    extent->sectors = sectors;

    extent->l1_table_offset = l1_offset;

    extent->l1_backup_table_offset = l1_backup_offset;

    extent->l1_size = l1_size;

    extent->l1_entry_sectors = l2_size * cluster_sectors;

    extent->l2_size = l2_size;

    extent->cluster_sectors = flat ? sectors : cluster_sectors;



    if (!flat) {

        bs->bl.write_zeroes_alignment =

            MAX(bs->bl.write_zeroes_alignment, cluster_sectors);

    }

    if (s->num_extents > 1) {

        extent->end_sector = (*(extent - 1)).end_sector + extent->sectors;

    } else {

        extent->end_sector = extent->sectors;

    }

    bs->total_sectors = extent->end_sector;

    if (new_extent) {

        *new_extent = extent;

    }

    return 0;

}
