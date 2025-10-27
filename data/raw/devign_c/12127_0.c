static void scsi_dma_restart_cb(void *opaque, int running, int reason)

{

    SCSIDeviceState *s = opaque;

    SCSIRequest *r = s->requests;

    if (!running)

        return;



    while (r) {

        if (r->status & SCSI_REQ_STATUS_RETRY) {

            r->status &= ~SCSI_REQ_STATUS_RETRY;

            scsi_write_request(r); 

        }

        r = r->next;

    }

}
