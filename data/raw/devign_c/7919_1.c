static int flush_blks(QEMUFile *f)

{

    BlkMigBlock *blk;

    int ret = 0;



    DPRINTF("%s Enter submitted %d read_done %d transferred %d\n",

            __FUNCTION__, block_mig_state.submitted, block_mig_state.read_done,

            block_mig_state.transferred);



    blk_mig_lock();

    while ((blk = QSIMPLEQ_FIRST(&block_mig_state.blk_list)) != NULL) {

        if (qemu_file_rate_limit(f)) {

            break;

        }

        if (blk->ret < 0) {

            ret = blk->ret;

            break;

        }



        QSIMPLEQ_REMOVE_HEAD(&block_mig_state.blk_list, entry);

        blk_mig_unlock();

        blk_send(f, blk);

        blk_mig_lock();



        g_free(blk->buf);

        g_free(blk);



        block_mig_state.read_done--;

        block_mig_state.transferred++;

        assert(block_mig_state.read_done >= 0);

    }

    blk_mig_unlock();



    DPRINTF("%s Exit submitted %d read_done %d transferred %d\n", __FUNCTION__,

            block_mig_state.submitted, block_mig_state.read_done,

            block_mig_state.transferred);

    return ret;

}
