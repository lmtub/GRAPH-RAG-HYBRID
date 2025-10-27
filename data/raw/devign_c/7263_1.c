static int qcow2_save_vmstate(BlockDriverState *bs, QEMUIOVector *qiov,

                              int64_t pos)

{

    BDRVQcowState *s = bs->opaque;

    int64_t total_sectors = bs->total_sectors;

    int growable = bs->growable;


    int ret;



    BLKDBG_EVENT(bs->file, BLKDBG_VMSTATE_SAVE);

    bs->growable = 1;


    ret = bdrv_pwritev(bs, qcow2_vm_state_offset(s) + pos, qiov);

    bs->growable = growable;




    /* bdrv_co_do_writev will have increased the total_sectors value to include

     * the VM state - the VM state is however not an actual part of the block

     * device, therefore, we need to restore the old value. */

    bs->total_sectors = total_sectors;



    return ret;

}