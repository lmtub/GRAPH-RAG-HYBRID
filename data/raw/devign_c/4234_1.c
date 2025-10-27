static int64_t get_remaining_dirty(void)

{

    BlkMigDevState *bmds;

    int64_t dirty = 0;



    QSIMPLEQ_FOREACH(bmds, &block_mig_state.bmds_list, entry) {

        dirty += bdrv_get_dirty_count(bmds->bs, bmds->dirty_bitmap);

    }



    return dirty << BDRV_SECTOR_BITS;

}
