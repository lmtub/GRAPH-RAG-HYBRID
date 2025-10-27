static void blk_mig_reset_dirty_cursor(void)

{

    BlkMigDevState *bmds;



    QSIMPLEQ_FOREACH(bmds, &block_mig_state.bmds_list, entry) {

        bmds->cur_dirty = 0;

    }

}
