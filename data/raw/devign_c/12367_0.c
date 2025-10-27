iscsi_unmap_cb(struct iscsi_context *iscsi, int status,

                     void *command_data, void *opaque)

{

    IscsiAIOCB *acb = opaque;



    if (acb->canceled != 0) {

        return;

    }



    acb->status = 0;

    if (status < 0) {

        error_report("Failed to unmap data on iSCSI lun. %s",

                     iscsi_get_error(iscsi));

        acb->status = -EIO;

    }



    iscsi_schedule_bh(acb);

}
