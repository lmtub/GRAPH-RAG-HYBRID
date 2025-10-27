static SCSIDiskReq *scsi_new_request(SCSIDiskState *s, uint32_t tag,

        uint32_t lun)

{

    SCSIRequest *req;

    SCSIDiskReq *r;



    req = scsi_req_alloc(sizeof(SCSIDiskReq), &s->qdev, tag, lun);

    r = DO_UPCAST(SCSIDiskReq, req, req);

    r->iov.iov_base = qemu_blockalign(s->bs, SCSI_DMA_BUF_SIZE);

    return r;

}
