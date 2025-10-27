static struct scsi_task *iscsi_do_inquiry(struct iscsi_context *iscsi, int lun,

                                          int evpd, int pc, void **inq, Error **errp)

{

    int full_size;

    struct scsi_task *task = NULL;

    task = iscsi_inquiry_sync(iscsi, lun, evpd, pc, 64);

    if (task == NULL || task->status != SCSI_STATUS_GOOD) {

        goto fail;

    }

    full_size = scsi_datain_getfullsize(task);

    if (full_size > task->datain.size) {

        scsi_free_scsi_task(task);



        /* we need more data for the full list */

        task = iscsi_inquiry_sync(iscsi, lun, evpd, pc, full_size);

        if (task == NULL || task->status != SCSI_STATUS_GOOD) {

            goto fail;

        }

    }



    *inq = scsi_datain_unmarshall(task);

    if (*inq == NULL) {

        error_setg(errp, "iSCSI: failed to unmarshall inquiry datain blob");

        goto fail;

    }



    return task;



fail:

    if (!error_is_set(errp)) {

        error_setg(errp, "iSCSI: Inquiry command failed : %s",

                   iscsi_get_error(iscsi));

    }

    if (task != NULL) {

        scsi_free_scsi_task(task);

    }

    return NULL;

}
