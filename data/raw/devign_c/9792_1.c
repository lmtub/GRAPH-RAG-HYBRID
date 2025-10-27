static SCSIDiskReq *scsi_find_request(SCSIDiskState *s, uint32_t tag)

{

    return DO_UPCAST(SCSIDiskReq, req, scsi_req_find(&s->qdev, tag));

}
