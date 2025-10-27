static int coroutine_fn iscsi_co_flush(BlockDriverState *bs)

{

    IscsiLun *iscsilun = bs->opaque;

    struct IscsiTask iTask;



    if (bdrv_is_sg(bs)) {

        return 0;

    }



    if (!iscsilun->force_next_flush) {

        return 0;

    }

    iscsilun->force_next_flush = false;



    iscsi_co_init_iscsitask(iscsilun, &iTask);

retry:

    if (iscsi_synchronizecache10_task(iscsilun->iscsi, iscsilun->lun, 0, 0, 0,

                                      0, iscsi_co_generic_cb, &iTask) == NULL) {

        return -ENOMEM;

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

        iTask.complete = 0;

        goto retry;

    }



    if (iTask.status != SCSI_STATUS_GOOD) {

        return -EIO;

    }



    return 0;

}
