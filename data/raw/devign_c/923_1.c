static int64_t coroutine_fn iscsi_co_get_block_status(BlockDriverState *bs,

                                                  int64_t sector_num,

                                                  int nb_sectors, int *pnum,

                                                  BlockDriverState **file)

{

    IscsiLun *iscsilun = bs->opaque;

    struct scsi_get_lba_status *lbas = NULL;

    struct scsi_lba_status_descriptor *lbasd = NULL;

    struct IscsiTask iTask;

    uint64_t lba;

    int64_t ret;



    if (!is_sector_request_lun_aligned(sector_num, nb_sectors, iscsilun)) {

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



    lba = sector_qemu2lun(sector_num, iscsilun);



    iscsi_co_init_iscsitask(iscsilun, &iTask);

    qemu_mutex_lock(&iscsilun->mutex);

retry:

    if (iscsi_get_lba_status_task(iscsilun->iscsi, iscsilun->lun,

                                  lba, 8 + 16, iscsi_co_generic_cb,

                                  &iTask) == NULL) {

        ret = -ENOMEM;

        goto out_unlock;

    }



    while (!iTask.complete) {

        iscsi_set_events(iscsilun);

        qemu_mutex_unlock(&iscsilun->mutex);

        qemu_coroutine_yield();

        qemu_mutex_lock(&iscsilun->mutex);

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

        error_report("iSCSI GET_LBA_STATUS failed at lba %" PRIu64 ": %s",

                     lba, iTask.err_str);

        goto out_unlock;

    }



    lbas = scsi_datain_unmarshall(iTask.task);

    if (lbas == NULL) {

        ret = -EIO;

        goto out_unlock;

    }



    lbasd = &lbas->descriptors[0];



    if (sector_qemu2lun(sector_num, iscsilun) != lbasd->lba) {

        ret = -EIO;

        goto out_unlock;

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

        iscsi_allocmap_set_unallocated(iscsilun, sector_num, *pnum);

    } else {

        iscsi_allocmap_set_allocated(iscsilun, sector_num, *pnum);

    }



    if (*pnum > nb_sectors) {

        *pnum = nb_sectors;

    }

out_unlock:

    qemu_mutex_unlock(&iscsilun->mutex);

    g_free(iTask.err_str);

out:

    if (iTask.task != NULL) {

        scsi_free_scsi_task(iTask.task);

    }

    if (ret > 0 && ret & BDRV_BLOCK_OFFSET_VALID) {

        *file = bs;

    }

    return ret;

}
