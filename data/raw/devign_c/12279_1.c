static int32_t scsi_send_command(SCSIRequest *req, uint8_t *cmd)

{

    SCSIGenericState *s = DO_UPCAST(SCSIGenericState, qdev, req->dev);

    SCSIGenericReq *r = DO_UPCAST(SCSIGenericReq, req, req);

    int ret;



    scsi_req_enqueue(req);

    if (cmd[0] != REQUEST_SENSE &&

        (req->lun != s->lun || (cmd[1] >> 5) != s->lun)) {

        DPRINTF("Unimplemented LUN %d\n", req->lun ? req->lun : cmd[1] >> 5);



        s->sensebuf[0] = 0x70;

        s->sensebuf[1] = 0x00;

        s->sensebuf[2] = ILLEGAL_REQUEST;

        s->sensebuf[3] = 0x00;

        s->sensebuf[4] = 0x00;

        s->sensebuf[5] = 0x00;

        s->sensebuf[6] = 0x00;

        s->senselen = 7;

        s->driver_status = SG_ERR_DRIVER_SENSE;

        r->req.status = CHECK_CONDITION;

        scsi_req_complete(&r->req);

        return 0;

    }



    if (-1 == scsi_req_parse(&r->req, cmd)) {

        BADF("Unsupported command length, command %x\n", cmd[0]);

        scsi_req_dequeue(&r->req);

        scsi_req_unref(&r->req);

        return 0;

    }

    scsi_req_fixup(&r->req);



    DPRINTF("Command: lun=%d tag=0x%x len %zd data=0x%02x", lun, tag,

            r->req.cmd.xfer, cmd[0]);



#ifdef DEBUG_SCSI

    {

        int i;

        for (i = 1; i < r->req.cmd.len; i++) {

            printf(" 0x%02x", cmd[i]);

        }

        printf("\n");

    }

#endif



    if (r->req.cmd.xfer == 0) {

        if (r->buf != NULL)

            qemu_free(r->buf);

        r->buflen = 0;

        r->buf = NULL;

        ret = execute_command(s->bs, r, SG_DXFER_NONE, scsi_command_complete);

        if (ret == -1) {

            scsi_command_complete(r, -EINVAL);

        }

        return 0;

    }



    if (r->buflen != r->req.cmd.xfer) {

        if (r->buf != NULL)

            qemu_free(r->buf);

        r->buf = qemu_malloc(r->req.cmd.xfer);

        r->buflen = r->req.cmd.xfer;

    }



    memset(r->buf, 0, r->buflen);

    r->len = r->req.cmd.xfer;

    if (r->req.cmd.mode == SCSI_XFER_TO_DEV) {

        r->len = 0;

        return -r->req.cmd.xfer;

    } else {

        return r->req.cmd.xfer;

    }

}
