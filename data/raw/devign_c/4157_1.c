static int scsi_disk_emulate_mode_sense(SCSIDiskReq *r, uint8_t *outbuf)

{

    SCSIDiskState *s = DO_UPCAST(SCSIDiskState, qdev, r->req.dev);

    uint64_t nb_sectors;

    int page, dbd, buflen, ret, page_control;

    uint8_t *p;

    uint8_t dev_specific_param;



    dbd = r->req.cmd.buf[1]  & 0x8;

    page = r->req.cmd.buf[2] & 0x3f;

    page_control = (r->req.cmd.buf[2] & 0xc0) >> 6;

    DPRINTF("Mode Sense(%d) (page %d, xfer %zd, page_control %d)\n",

        (r->req.cmd.buf[0] == MODE_SENSE) ? 6 : 10, page, r->req.cmd.xfer, page_control);

    memset(outbuf, 0, r->req.cmd.xfer);

    p = outbuf;



    if (bdrv_is_read_only(s->qdev.conf.bs)) {

        dev_specific_param = 0x80; /* Readonly.  */

    } else {

        dev_specific_param = 0x00;

    }



    if (r->req.cmd.buf[0] == MODE_SENSE) {

        p[1] = 0; /* Default media type.  */

        p[2] = dev_specific_param;

        p[3] = 0; /* Block descriptor length.  */

        p += 4;

    } else { /* MODE_SENSE_10 */

        p[2] = 0; /* Default media type.  */

        p[3] = dev_specific_param;

        p[6] = p[7] = 0; /* Block descriptor length.  */

        p += 8;

    }



    /* MMC prescribes that CD/DVD drives have no block descriptors.  */

    bdrv_get_geometry(s->qdev.conf.bs, &nb_sectors);

    if (!dbd && s->qdev.type == TYPE_DISK && nb_sectors) {

        if (r->req.cmd.buf[0] == MODE_SENSE) {

            outbuf[3] = 8; /* Block descriptor length  */

        } else { /* MODE_SENSE_10 */

            outbuf[7] = 8; /* Block descriptor length  */

        }

        nb_sectors /= (s->qdev.blocksize / 512);

        if (nb_sectors > 0xffffff) {

            nb_sectors = 0;

        }

        p[0] = 0; /* media density code */

        p[1] = (nb_sectors >> 16) & 0xff;

        p[2] = (nb_sectors >> 8) & 0xff;

        p[3] = nb_sectors & 0xff;

        p[4] = 0; /* reserved */

        p[5] = 0; /* bytes 5-7 are the sector size in bytes */

        p[6] = s->qdev.blocksize >> 8;

        p[7] = 0;

        p += 8;

    }



    if (page_control == 3) {

        /* Saved Values */

        scsi_check_condition(r, SENSE_CODE(SAVING_PARAMS_NOT_SUPPORTED));

        return -1;

    }



    if (page == 0x3f) {

        for (page = 0; page <= 0x3e; page++) {

            mode_sense_page(s, page, &p, page_control);

        }

    } else {

        ret = mode_sense_page(s, page, &p, page_control);

        if (ret == -1) {

            return -1;

        }

    }



    buflen = p - outbuf;

    /*

     * The mode data length field specifies the length in bytes of the

     * following data that is available to be transferred. The mode data

     * length does not include itself.

     */

    if (r->req.cmd.buf[0] == MODE_SENSE) {

        outbuf[0] = buflen - 1;

    } else { /* MODE_SENSE_10 */

        outbuf[0] = ((buflen - 2) >> 8) & 0xff;

        outbuf[1] = (buflen - 2) & 0xff;

    }

    if (buflen > r->req.cmd.xfer) {

        buflen = r->req.cmd.xfer;

    }

    return buflen;

}
