void scsi_req_build_sense(SCSIRequest *req, SCSISense sense)

{

    trace_scsi_req_build_sense(req->dev->id, req->lun, req->tag,

                               sense.key, sense.asc, sense.ascq);

    memset(req->sense, 0, 18);

    req->sense[0] = 0xf0;

    req->sense[2] = sense.key;

    req->sense[7] = 10;

    req->sense[12] = sense.asc;

    req->sense[13] = sense.ascq;

    req->sense_len = 18;

}
