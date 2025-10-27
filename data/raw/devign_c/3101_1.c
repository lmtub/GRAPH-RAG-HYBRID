iscsi_synccache10_cb(struct iscsi_context *iscsi, int status,

                     void *command_data, void *opaque)

{

    IscsiAIOCB *acb = opaque;



    if (acb->canceled != 0) {

        qemu_aio_release(acb);

        scsi_free_scsi_task(acb->task);

        acb->task = NULL;

        return;

    }



    acb->status = 0;

    if (status < 0) {

        error_report("Failed to sync10 data on iSCSI lun. %s",

                     iscsi_get_error(iscsi));

        acb->status = -EIO;

    }



    iscsi_schedule_bh(acb);

    scsi_free_scsi_task(acb->task);

    acb->task = NULL;

}
