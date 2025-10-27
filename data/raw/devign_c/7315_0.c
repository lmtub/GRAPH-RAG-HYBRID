static int32_t scsi_send_command(SCSIRequest *req, uint8_t *cmd)

{

    SCSIGenericState *s = DO_UPCAST(SCSIGenericState, qdev, req->dev);

    SCSIGenericReq *r = DO_UPCAST(SCSIGenericReq, req, req);

    int ret;



    if (cmd[0] != REQUEST_SENSE && req->lun != s->qdev.lun) {

        DPRINTF("Unimplemented LUN %d\n", req->lun);

        scsi_req_build_sense(&r->req, SENSE_CODE(LUN_NOT_SUPPORTED));

        scsi_req_complete(&r->req, CHECK_CONDITION);

        return 0;

    }



    if (-1 == scsi_req_parse(&r->req, cmd)) {

        BADF("Unsupported command length, command %x\n", cmd[0]);

        scsi_command_complete(r, -EINVAL);

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

        if (ret < 0) {

            scsi_command_complete(r, ret);

            return 0;

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
