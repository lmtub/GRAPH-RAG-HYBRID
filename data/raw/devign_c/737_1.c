static BlockDriverAIOCB *iscsi_aio_ioctl(BlockDriverState *bs,

        unsigned long int req, void *buf,

        BlockDriverCompletionFunc *cb, void *opaque)

{

    IscsiLun *iscsilun = bs->opaque;

    struct iscsi_context *iscsi = iscsilun->iscsi;

    struct iscsi_data data;

    IscsiAIOCB *acb;



    assert(req == SG_IO);



    acb = qemu_aio_get(&iscsi_aiocb_info, bs, cb, opaque);



    acb->iscsilun = iscsilun;

    acb->canceled    = 0;

    acb->bh          = NULL;

    acb->status      = -EINPROGRESS;

    acb->buf         = NULL;

    acb->ioh         = buf;



    acb->task = malloc(sizeof(struct scsi_task));

    if (acb->task == NULL) {

        error_report("iSCSI: Failed to allocate task for scsi command. %s",

                     iscsi_get_error(iscsi));

        qemu_aio_release(acb);

        return NULL;

    }

    memset(acb->task, 0, sizeof(struct scsi_task));



    switch (acb->ioh->dxfer_direction) {

    case SG_DXFER_TO_DEV:

        acb->task->xfer_dir = SCSI_XFER_WRITE;

        break;

    case SG_DXFER_FROM_DEV:

        acb->task->xfer_dir = SCSI_XFER_READ;

        break;

    default:

        acb->task->xfer_dir = SCSI_XFER_NONE;

        break;

    }



    acb->task->cdb_size = acb->ioh->cmd_len;

    memcpy(&acb->task->cdb[0], acb->ioh->cmdp, acb->ioh->cmd_len);

    acb->task->expxferlen = acb->ioh->dxfer_len;



    data.size = 0;

    if (acb->task->xfer_dir == SCSI_XFER_WRITE) {

        if (acb->ioh->iovec_count == 0) {

            data.data = acb->ioh->dxferp;

            data.size = acb->ioh->dxfer_len;

        } else {

#if defined(LIBISCSI_FEATURE_IOVECTOR)

            scsi_task_set_iov_out(acb->task,

                                 (struct scsi_iovec *) acb->ioh->dxferp,

                                 acb->ioh->iovec_count);

#else

            struct iovec *iov = (struct iovec *)acb->ioh->dxferp;



            acb->buf = g_malloc(acb->ioh->dxfer_len);

            data.data = acb->buf;

            data.size = iov_to_buf(iov, acb->ioh->iovec_count, 0,

                                   acb->buf, acb->ioh->dxfer_len);

#endif

        }

    }



    if (iscsi_scsi_command_async(iscsi, iscsilun->lun, acb->task,

                                 iscsi_aio_ioctl_cb,

                                 (data.size > 0) ? &data : NULL,

                                 acb) != 0) {

        scsi_free_scsi_task(acb->task);

        qemu_aio_release(acb);

        return NULL;

    }



    /* tell libiscsi to read straight into the buffer we got from ioctl */

    if (acb->task->xfer_dir == SCSI_XFER_READ) {

        if (acb->ioh->iovec_count == 0) {

            scsi_task_add_data_in_buffer(acb->task,

                                         acb->ioh->dxfer_len,

                                         acb->ioh->dxferp);

        } else {

#if defined(LIBISCSI_FEATURE_IOVECTOR)

            scsi_task_set_iov_in(acb->task,

                                 (struct scsi_iovec *) acb->ioh->dxferp,

                                 acb->ioh->iovec_count);

#else

            int i;

            for (i = 0; i < acb->ioh->iovec_count; i++) {

                struct iovec *iov = (struct iovec *)acb->ioh->dxferp;



                scsi_task_add_data_in_buffer(acb->task,

                    iov[i].iov_len,

                    iov[i].iov_base);

            }

#endif

        }

    }



    iscsi_set_events(iscsilun);



    return &acb->common;

}
