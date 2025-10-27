static SCSIRequest *scsi_new_request(SCSIDevice *d, uint32_t tag, uint32_t lun,

                                     uint8_t *buf, void *hba_private)

{

    SCSIRequest *req;



    req = scsi_req_alloc(&scsi_generic_req_ops, d, tag, lun, hba_private);

    return req;

}
