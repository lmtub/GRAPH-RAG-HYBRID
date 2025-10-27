iscsi_aio_discard(BlockDriverState *bs,

                  int64_t sector_num, int nb_sectors,

                  BlockDriverCompletionFunc *cb, void *opaque)

{

    IscsiLun *iscsilun = bs->opaque;

    struct iscsi_context *iscsi = iscsilun->iscsi;

    IscsiAIOCB *acb;

    struct unmap_list list[1];



    acb = qemu_aio_get(&iscsi_aiocb_info, bs, cb, opaque);



    acb->iscsilun = iscsilun;

    acb->canceled   = 0;

    acb->bh         = NULL;

    acb->status     = -EINPROGRESS;

    acb->buf        = NULL;



    list[0].lba = sector_qemu2lun(sector_num, iscsilun);

    list[0].num = nb_sectors * BDRV_SECTOR_SIZE / iscsilun->block_size;



    acb->task = iscsi_unmap_task(iscsi, iscsilun->lun,

                                 0, 0, &list[0], 1,

                                 iscsi_unmap_cb,

                                 acb);

    if (acb->task == NULL) {

        error_report("iSCSI: Failed to send unmap command. %s",

                     iscsi_get_error(iscsi));

        qemu_aio_release(acb);

        return NULL;

    }



    iscsi_set_events(iscsilun);



    return &acb->common;

}
