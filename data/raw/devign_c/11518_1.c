static SCSIGenericReq *scsi_find_request(SCSIGenericState *s, uint32_t tag)

{

    return DO_UPCAST(SCSIGenericReq, req, scsi_req_find(&s->qdev, tag));

}
