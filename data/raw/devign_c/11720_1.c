static void blk_mig_cleanup(void)

{

    BlkMigDevState *bmds;

    BlkMigBlock *blk;



    bdrv_drain_all();



    unset_dirty_tracking();



    blk_mig_lock();

    while ((bmds = QSIMPLEQ_FIRST(&block_mig_state.bmds_list)) != NULL) {

        QSIMPLEQ_REMOVE_HEAD(&block_mig_state.bmds_list, entry);

        bdrv_op_unblock_all(bmds->bs, bmds->blocker);

        error_free(bmds->blocker);

        bdrv_unref(bmds->bs);

        g_free(bmds->aio_bitmap);

        g_free(bmds);

    }



    while ((blk = QSIMPLEQ_FIRST(&block_mig_state.blk_list)) != NULL) {

        QSIMPLEQ_REMOVE_HEAD(&block_mig_state.blk_list, entry);

        g_free(blk->buf);

        g_free(blk);

    }

    blk_mig_unlock();

}
