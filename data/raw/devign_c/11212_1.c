static void unset_dirty_tracking(void)

{

    BlkMigDevState *bmds;



    QSIMPLEQ_FOREACH(bmds, &block_mig_state.bmds_list, entry) {

        bdrv_release_dirty_bitmap(bmds->bs, bmds->dirty_bitmap);

    }

}
