static int64_t coroutine_fn iscsi_co_get_block_status(BlockDriverState *bs,

                                                  int64_t sector_num,

                                                  int nb_sectors, int *pnum)

{

    IscsiLun *iscsilun = bs->opaque;

    struct scsi_get_lba_status *lbas = NULL;

    struct scsi_lba_status_descriptor *lbasd = NULL;

    struct IscsiTask iTask;

    int64_t ret;



    iscsi_co_init_iscsitask(iscsilun, &iTask);



    if (!is_request_lun_aligned(sector_num, nb_sectors, iscsilun)) {

        ret = -EINVAL;

        goto out;

    }



    /* default to all sectors allocated */

    ret = BDRV_BLOCK_DATA;

    ret |= (sector_num << BDRV_SECTOR_BITS) | BDRV_BLOCK_OFFSET_VALID;

    *pnum = nb_sectors;



    /* LUN does not support logical block provisioning */

    if (!iscsilun->lbpme) {

        goto out;

    }



retry:

    if (iscsi_get_lba_status_task(iscsilun->iscsi, iscsilun->lun,

                                  sector_qemu2lun(sector_num, iscsilun),

                                  8 + 16, iscsi_co_generic_cb,

                                  &iTask) == NULL) {

        ret = -ENOMEM;

        goto out;

    }



    while (!iTask.complete) {

        iscsi_set_events(iscsilun);

        qemu_coroutine_yield();

    }



    if (iTask.do_retry) {

        if (iTask.task != NULL) {

            scsi_free_scsi_task(iTask.task);

            iTask.task = NULL;

        }

        iTask.complete = 0;

        goto retry;

    }



    if (iTask.status != SCSI_STATUS_GOOD) {

        /* in case the get_lba_status_callout fails (i.e.

         * because the device is busy or the cmd is not

         * supported) we pretend all blocks are allocated

         * for backwards compatibility */

        goto out;

    }



    lbas = scsi_datain_unmarshall(iTask.task);

    if (lbas == NULL) {

        ret = -EIO;

        goto out;

    }



    lbasd = &lbas->descriptors[0];



    if (sector_qemu2lun(sector_num, iscsilun) != lbasd->lba) {

        ret = -EIO;

        goto out;

    }



    *pnum = sector_lun2qemu(lbasd->num_blocks, iscsilun);



    if (lbasd->provisioning == SCSI_PROVISIONING_TYPE_DEALLOCATED ||

        lbasd->provisioning == SCSI_PROVISIONING_TYPE_ANCHORED) {

        ret &= ~BDRV_BLOCK_DATA;

        if (iscsilun->lbprz) {

            ret |= BDRV_BLOCK_ZERO;

        }

    }



    if (ret & BDRV_BLOCK_ZERO) {

        iscsi_allocationmap_clear(iscsilun, sector_num, *pnum);

    } else {

        iscsi_allocationmap_set(iscsilun, sector_num, *pnum);

    }



    if (*pnum > nb_sectors) {

        *pnum = nb_sectors;

    }

out:

    if (iTask.task != NULL) {

        scsi_free_scsi_task(iTask.task);

    }

    return ret;

}
