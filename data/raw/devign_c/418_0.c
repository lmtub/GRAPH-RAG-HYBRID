static void unset_dirty_tracking(void)

{

    BlkMigDevState *bmds;



    QSIMPLEQ_FOREACH(bmds, &block_mig_state.bmds_list, entry) {

        aio_context_acquire(blk_get_aio_context(bmds->blk));

        bdrv_release_dirty_bitmap(blk_bs(bmds->blk), bmds->dirty_bitmap);

        aio_context_release(blk_get_aio_context(bmds->blk));

    }

}
