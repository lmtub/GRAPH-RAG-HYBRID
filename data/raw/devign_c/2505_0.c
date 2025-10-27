iscsi_aio_writev(BlockDriverState *bs, int64_t sector_num,

                 QEMUIOVector *qiov, int nb_sectors,

                 BlockDriverCompletionFunc *cb,

                 void *opaque)

{

    IscsiLun *iscsilun = bs->opaque;

    struct iscsi_context *iscsi = iscsilun->iscsi;

    IscsiAIOCB *acb;

    size_t size;

    uint32_t num_sectors;

    uint64_t lba;

#if !defined(LIBISCSI_FEATURE_IOVECTOR)

    struct iscsi_data data;

#endif

    int ret;



    acb = qemu_aio_get(&iscsi_aiocb_info, bs, cb, opaque);

    trace_iscsi_aio_writev(iscsi, sector_num, nb_sectors, opaque, acb);



    acb->iscsilun = iscsilun;

    acb->qiov     = qiov;



    acb->canceled   = 0;

    acb->bh         = NULL;

    acb->status     = -EINPROGRESS;

    acb->buf        = NULL;



    /* this will allow us to get rid of 'buf' completely */

    size = nb_sectors * BDRV_SECTOR_SIZE;



#if !defined(LIBISCSI_FEATURE_IOVECTOR)

    data.size = MIN(size, acb->qiov->size);



    /* if the iovec only contains one buffer we can pass it directly */

    if (acb->qiov->niov == 1) {

        data.data = acb->qiov->iov[0].iov_base;

    } else {

        acb->buf = g_malloc(data.size);

        qemu_iovec_to_buf(acb->qiov, 0, acb->buf, data.size);

        data.data = acb->buf;

    }

#endif



    acb->task = malloc(sizeof(struct scsi_task));

    if (acb->task == NULL) {

        error_report("iSCSI: Failed to allocate task for scsi WRITE16 "

                     "command. %s", iscsi_get_error(iscsi));

        qemu_aio_release(acb);

        return NULL;

    }

    memset(acb->task, 0, sizeof(struct scsi_task));



    acb->task->xfer_dir = SCSI_XFER_WRITE;

    acb->task->cdb_size = 16;

    acb->task->cdb[0] = 0x8a;

    lba = sector_qemu2lun(sector_num, iscsilun);

    *(uint32_t *)&acb->task->cdb[2]  = htonl(lba >> 32);

    *(uint32_t *)&acb->task->cdb[6]  = htonl(lba & 0xffffffff);

    num_sectors = size / iscsilun->block_size;

    *(uint32_t *)&acb->task->cdb[10] = htonl(num_sectors);

    acb->task->expxferlen = size;



#if defined(LIBISCSI_FEATURE_IOVECTOR)

    ret = iscsi_scsi_command_async(iscsi, iscsilun->lun, acb->task,

                                   iscsi_aio_write16_cb,

                                   NULL,

                                   acb);

#else

    ret = iscsi_scsi_command_async(iscsi, iscsilun->lun, acb->task,

                                   iscsi_aio_write16_cb,

                                   &data,

                                   acb);

#endif

    if (ret != 0) {

        scsi_free_scsi_task(acb->task);

        g_free(acb->buf);

        qemu_aio_release(acb);

        return NULL;

    }



#if defined(LIBISCSI_FEATURE_IOVECTOR)

    scsi_task_set_iov_out(acb->task, (struct scsi_iovec*) acb->qiov->iov, acb->qiov->niov);

#endif



    iscsi_set_events(iscsilun);



    return &acb->common;

}
