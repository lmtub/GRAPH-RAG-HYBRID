static int32_t scsi_send_command(SCSIDevice *d, uint32_t tag,

                                 uint8_t *cmd, int lun)

{

    SCSIGenericState *s = DO_UPCAST(SCSIGenericState, qdev, d);

    SCSIGenericReq *r;

    SCSIBus *bus;

    int ret;



    if (cmd[0] != REQUEST_SENSE &&

        (lun != s->lun || (cmd[1] >> 5) != s->lun)) {

        DPRINTF("Unimplemented LUN %d\n", lun ? lun : cmd[1] >> 5);



        s->sensebuf[0] = 0x70;

        s->sensebuf[1] = 0x00;

        s->sensebuf[2] = ILLEGAL_REQUEST;

        s->sensebuf[3] = 0x00;

        s->sensebuf[4] = 0x00;

        s->sensebuf[5] = 0x00;

        s->sensebuf[6] = 0x00;

        s->senselen = 7;

        s->driver_status = SG_ERR_DRIVER_SENSE;

        bus = scsi_bus_from_device(d);

        bus->ops->complete(bus, SCSI_REASON_DONE, tag, CHECK_CONDITION);

        return 0;

    }



    r = scsi_find_request(s, tag);

    if (r) {

        BADF("Tag 0x%x already in use %p\n", tag, r);

        scsi_cancel_io(d, tag);

    }

    r = scsi_new_request(d, tag, lun);



    if (-1 == scsi_req_parse(&r->req, cmd)) {

        BADF("Unsupported command length, command %x\n", cmd[0]);

        scsi_remove_request(r);

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

    }



    return r->req.cmd.xfer;

}
