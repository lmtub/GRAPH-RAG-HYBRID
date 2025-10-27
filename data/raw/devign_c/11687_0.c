static int scsi_event_status_media(SCSIDiskState *s, uint8_t *outbuf)

{

    uint8_t event_code, media_status;



    media_status = 0;

    if (s->tray_open) {

        media_status = MS_TRAY_OPEN;

    } else if (bdrv_is_inserted(s->qdev.conf.bs)) {

        media_status = MS_MEDIA_PRESENT;

    }



    /* Event notification descriptor */

    event_code = MEC_NO_CHANGE;

    if (media_status != MS_TRAY_OPEN) {

        if (s->media_event) {

            event_code = MEC_NEW_MEDIA;

            s->media_event = false;

        } else if (s->eject_request) {

            event_code = MEC_EJECT_REQUESTED;

            s->eject_request = false;

        }

    }



    outbuf[0] = event_code;

    outbuf[1] = media_status;



    /* These fields are reserved, just clear them. */

    outbuf[2] = 0;

    outbuf[3] = 0;

    return 4;

}
