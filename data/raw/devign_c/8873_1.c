coroutine_fn iscsi_co_write_zeroes(BlockDriverState *bs, int64_t sector_num,

                                   int nb_sectors, BdrvRequestFlags flags)

{

    IscsiLun *iscsilun = bs->opaque;

    struct IscsiTask iTask;

    uint64_t lba;

    uint32_t nb_blocks;



    if (!is_request_lun_aligned(sector_num, nb_sectors, iscsilun)) {

        return -EINVAL;

    }



    if (!iscsilun->lbp.lbpws) {

        /* WRITE SAME is not supported by the target */

        return -ENOTSUP;

    }



    lba = sector_qemu2lun(sector_num, iscsilun);

    nb_blocks = sector_qemu2lun(nb_sectors, iscsilun);



    if (iscsilun->zeroblock == NULL) {

        iscsilun->zeroblock = g_malloc0(iscsilun->block_size);

    }



    iscsi_co_init_iscsitask(iscsilun, &iTask);

retry:

    if (iscsi_writesame16_task(iscsilun->iscsi, iscsilun->lun, lba,

                               iscsilun->zeroblock, iscsilun->block_size,

                               nb_blocks, 0, !!(flags & BDRV_REQ_MAY_UNMAP),

                               0, 0, iscsi_co_generic_cb, &iTask) == NULL) {

        return -EIO;

    }



    while (!iTask.complete) {

        iscsi_set_events(iscsilun);

        qemu_coroutine_yield();

    }



    if (iTask.task != NULL) {

        scsi_free_scsi_task(iTask.task);

        iTask.task = NULL;

    }



    if (iTask.do_retry) {

        goto retry;

    }



    if (iTask.status != SCSI_STATUS_GOOD) {

        return -EIO;

    }



    return 0;

}
