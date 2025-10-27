iscsi_aio_ioctl_cb(struct iscsi_context *iscsi, int status,

                     void *command_data, void *opaque)

{

    IscsiAIOCB *acb = opaque;



    if (acb->canceled) {

        qemu_aio_release(acb);

        return;

    }



    acb->status = 0;

    if (status < 0) {

        error_report("Failed to ioctl(SG_IO) to iSCSI lun. %s",

                     iscsi_get_error(iscsi));

        acb->status = -EIO;

    }



    acb->ioh->driver_status = 0;

    acb->ioh->host_status   = 0;

    acb->ioh->resid         = 0;



#define SG_ERR_DRIVER_SENSE    0x08



    if (status == SCSI_STATUS_CHECK_CONDITION && acb->task->datain.size >= 2) {

        int ss;



        acb->ioh->driver_status |= SG_ERR_DRIVER_SENSE;



        acb->ioh->sb_len_wr = acb->task->datain.size - 2;

        ss = (acb->ioh->mx_sb_len >= acb->ioh->sb_len_wr) ?

             acb->ioh->mx_sb_len : acb->ioh->sb_len_wr;

        memcpy(acb->ioh->sbp, &acb->task->datain.data[2], ss);

    }



    iscsi_schedule_bh(iscsi_readv_writev_bh_cb, acb);

}
