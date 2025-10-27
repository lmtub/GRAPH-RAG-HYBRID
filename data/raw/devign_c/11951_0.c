int scsi_req_parse(SCSIRequest *req, uint8_t *buf)

{

    int rc;



    if (req->dev->type == TYPE_TAPE) {

        rc = scsi_req_stream_length(&req->cmd, req->dev, buf);

    } else {

        rc = scsi_req_length(&req->cmd, req->dev, buf);

    }

    if (rc != 0)

        return rc;



    assert(buf == req->cmd.buf);

    scsi_cmd_xfer_mode(&req->cmd);

    req->cmd.lba = scsi_cmd_lba(&req->cmd);

    trace_scsi_req_parsed(req->dev->id, req->lun, req->tag, buf[0],

                          req->cmd.mode, req->cmd.xfer);

    if (req->cmd.lba != -1) {

        trace_scsi_req_parsed_lba(req->dev->id, req->lun, req->tag, buf[0],

                              req->cmd.lba);

    }

    return 0;

}
