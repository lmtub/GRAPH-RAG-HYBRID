static void scsi_remove_request(SCSIGenericReq *r)

{

    qemu_free(r->buf);

    scsi_req_free(&r->req);

}
