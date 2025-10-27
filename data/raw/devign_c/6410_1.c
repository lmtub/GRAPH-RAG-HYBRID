iscsi_aio_readv(BlockDriverState *bs, int64_t sector_num,

                QEMUIOVector *qiov, int nb_sectors,

                BlockDriverCompletionFunc *cb,

                void *opaque)

{

    IscsiLun *iscsilun = bs->opaque;

    struct iscsi_context *iscsi = iscsilun->iscsi;

    IscsiAIOCB *acb;

    size_t qemu_read_size;

    int i;

    uint64_t lba;

    uint32_t num_sectors;



    qemu_read_size = BDRV_SECTOR_SIZE * (size_t)nb_sectors;



    acb = qemu_aio_get(&iscsi_aiocb_info, bs, cb, opaque);

    trace_iscsi_aio_readv(iscsi, sector_num, nb_sectors, opaque, acb);



    acb->iscsilun = iscsilun;

    acb->qiov     = qiov;



    acb->canceled    = 0;

    acb->bh          = NULL;

    acb->status      = -EINPROGRESS;

    acb->read_size   = qemu_read_size;

    acb->buf         = NULL;



    /* If LUN blocksize is bigger than BDRV_BLOCK_SIZE a read from QEMU

     * may be misaligned to the LUN, so we may need to read some extra

     * data.

     */

    acb->read_offset = 0;

    if (iscsilun->block_size > BDRV_SECTOR_SIZE) {

        uint64_t bdrv_offset = BDRV_SECTOR_SIZE * sector_num;



        acb->read_offset  = bdrv_offset % iscsilun->block_size;

    }



    num_sectors  = (qemu_read_size + iscsilun->block_size

                    + acb->read_offset - 1)

                    / iscsilun->block_size;



    acb->task = malloc(sizeof(struct scsi_task));

    if (acb->task == NULL) {

        error_report("iSCSI: Failed to allocate task for scsi READ16 "

                     "command. %s", iscsi_get_error(iscsi));

        qemu_aio_release(acb);

        return NULL;

    }

    memset(acb->task, 0, sizeof(struct scsi_task));



    acb->task->xfer_dir = SCSI_XFER_READ;

    lba = sector_qemu2lun(sector_num, iscsilun);

    acb->task->expxferlen = qemu_read_size;



    switch (iscsilun->type) {

    case TYPE_DISK:

        acb->task->cdb_size = 16;

        acb->task->cdb[0]  = 0x88;

        *(uint32_t *)&acb->task->cdb[2]  = htonl(lba >> 32);

        *(uint32_t *)&acb->task->cdb[6]  = htonl(lba & 0xffffffff);

        *(uint32_t *)&acb->task->cdb[10] = htonl(num_sectors);

        break;

    default:

        acb->task->cdb_size = 10;

        acb->task->cdb[0]  = 0x28;

        *(uint32_t *)&acb->task->cdb[2] = htonl(lba);

        *(uint16_t *)&acb->task->cdb[7] = htons(num_sectors);

        break;

    }

    

    if (iscsi_scsi_command_async(iscsi, iscsilun->lun, acb->task,

                                 iscsi_aio_read16_cb,

                                 NULL,

                                 acb) != 0) {

        scsi_free_scsi_task(acb->task);

        qemu_aio_release(acb);

        return NULL;

    }



    for (i = 0; i < acb->qiov->niov; i++) {

        scsi_task_add_data_in_buffer(acb->task,

                acb->qiov->iov[i].iov_len,

                acb->qiov->iov[i].iov_base);

    }



    iscsi_set_events(iscsilun);



    return &acb->common;

}
