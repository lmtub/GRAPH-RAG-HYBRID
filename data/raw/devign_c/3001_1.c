coroutine_fn iscsi_co_pdiscard(BlockDriverState *bs, int64_t offset, int bytes)

{

    IscsiLun *iscsilun = bs->opaque;

    struct IscsiTask iTask;

    struct unmap_list list;

    int r = 0;



    if (!is_byte_request_lun_aligned(offset, bytes, iscsilun)) {

        return -ENOTSUP;

    }



    if (!iscsilun->lbp.lbpu) {

        /* UNMAP is not supported by the target */

        return 0;

    }



    list.lba = offset / iscsilun->block_size;

    list.num = bytes / iscsilun->block_size;



    iscsi_co_init_iscsitask(iscsilun, &iTask);

    qemu_mutex_lock(&iscsilun->mutex);

retry:

    if (iscsi_unmap_task(iscsilun->iscsi, iscsilun->lun, 0, 0, &list, 1,

                         iscsi_co_generic_cb, &iTask) == NULL) {

        r = -ENOMEM;

        goto out_unlock;

    }



    while (!iTask.complete) {

        iscsi_set_events(iscsilun);

        qemu_mutex_unlock(&iscsilun->mutex);

        qemu_coroutine_yield();

        qemu_mutex_lock(&iscsilun->mutex);

    }



    if (iTask.task != NULL) {

        scsi_free_scsi_task(iTask.task);

        iTask.task = NULL;

    }



    if (iTask.do_retry) {

        iTask.complete = 0;

        goto retry;

    }



    if (iTask.status == SCSI_STATUS_CHECK_CONDITION) {

        /* the target might fail with a check condition if it

           is not happy with the alignment of the UNMAP request

           we silently fail in this case */

        goto out_unlock;

    }



    if (iTask.status != SCSI_STATUS_GOOD) {

        r = iTask.err_code;

        goto out_unlock;

    }



    iscsi_allocmap_set_invalid(iscsilun, offset >> BDRV_SECTOR_BITS,

                               bytes >> BDRV_SECTOR_BITS);



out_unlock:

    qemu_mutex_unlock(&iscsilun->mutex);

    return r;

}
