static VmdkExtent *vmdk_add_extent(BlockDriverState *bs,

                           BlockDriverState *file, bool flat, int64_t sectors,

                           int64_t l1_offset, int64_t l1_backup_offset,

                           uint32_t l1_size,

                           int l2_size, unsigned int cluster_sectors)

{

    VmdkExtent *extent;

    BDRVVmdkState *s = bs->opaque;



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

    extent->cluster_sectors = cluster_sectors;



    if (s->num_extents > 1) {

        extent->end_sector = (*(extent - 1)).end_sector + extent->sectors;

    } else {

        extent->end_sector = extent->sectors;

    }

    bs->total_sectors = extent->end_sector;

    return extent;

}
