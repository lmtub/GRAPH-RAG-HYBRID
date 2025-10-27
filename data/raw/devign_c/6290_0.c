static int block_save_complete(QEMUFile *f, void *opaque)

{

    int ret;



    DPRINTF("Enter save live complete submitted %d transferred %d\n",

            block_mig_state.submitted, block_mig_state.transferred);



    ret = flush_blks(f);

    if (ret) {

        blk_mig_cleanup();

        return ret;

    }



    blk_mig_reset_dirty_cursor();



    /* we know for sure that save bulk is completed and

       all async read completed */

    assert(block_mig_state.submitted == 0);



    do {

        ret = blk_mig_save_dirty_block(f, 0);

    } while (ret == 0);



    blk_mig_cleanup();

    if (ret) {

        return ret;

    }

    /* report completion */

    qemu_put_be64(f, (100 << BDRV_SECTOR_BITS) | BLK_MIG_FLAG_PROGRESS);



    DPRINTF("Block migration completed\n");



    qemu_put_be64(f, BLK_MIG_FLAG_EOS);



    return 0;

}
