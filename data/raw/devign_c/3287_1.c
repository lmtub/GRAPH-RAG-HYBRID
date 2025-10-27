static void scsi_remove_request(SCSIDiskReq *r)

{

    qemu_free(r->iov.iov_base);

    scsi_req_free(&r->req);

}
