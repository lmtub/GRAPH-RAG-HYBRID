static void scsi_cancel_io(SCSIDevice *d, uint32_t tag)

{

    DPRINTF("scsi_cancel_io 0x%x\n", tag);

    SCSIGenericState *s = DO_UPCAST(SCSIGenericState, qdev, d);

    SCSIGenericReq *r;

    DPRINTF("Cancel tag=0x%x\n", tag);

    r = scsi_find_request(s, tag);

    if (r) {

        if (r->req.aiocb)

            bdrv_aio_cancel(r->req.aiocb);

        r->req.aiocb = NULL;

        scsi_req_dequeue(&r->req);

    }

}
