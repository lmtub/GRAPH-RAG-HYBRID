static void set_dirty_tracking(void)

{

    BlkMigDevState *bmds;



    QSIMPLEQ_FOREACH(bmds, &block_mig_state.bmds_list, entry) {

        bmds->dirty_bitmap = bdrv_create_dirty_bitmap(bmds->bs, BLOCK_SIZE);

    }

}
