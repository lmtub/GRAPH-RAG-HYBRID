static int scsi_disk_emulate_start_stop(SCSIDiskReq *r)

{

    SCSIRequest *req = &r->req;

    SCSIDiskState *s = DO_UPCAST(SCSIDiskState, qdev, req->dev);

    bool start = req->cmd.buf[4] & 1;

    bool loej = req->cmd.buf[4] & 2; /* load on start, eject on !start */



    if (s->qdev.type == TYPE_ROM && loej) {

        if (!start && !s->tray_open && s->tray_locked) {

            scsi_check_condition(r,

                                 bdrv_is_inserted(s->qdev.conf.bs)

                                 ? SENSE_CODE(ILLEGAL_REQ_REMOVAL_PREVENTED)

                                 : SENSE_CODE(NOT_READY_REMOVAL_PREVENTED));

            return -1;

        }



        if (s->tray_open != !start) {

            bdrv_eject(s->qdev.conf.bs, !start);

            s->tray_open = !start;

        }

    }

    return 0;

}
