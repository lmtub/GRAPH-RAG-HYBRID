static int scsi_disk_emulate_mode_sense(SCSIRequest *req, uint8_t *outbuf)

{

    SCSIDiskState *s = DO_UPCAST(SCSIDiskState, qdev, req->dev);

    uint64_t nb_sectors;

    int page, dbd, buflen;

    uint8_t *p;

    uint8_t dev_specific_param;



    dbd = req->cmd.buf[1]  & 0x8;

    page = req->cmd.buf[2] & 0x3f;

    DPRINTF("Mode Sense (page %d, len %zd)\n", page, req->cmd.xfer);

    memset(outbuf, 0, req->cmd.xfer);

    p = outbuf;



    if (bdrv_is_read_only(s->bs)) {

        dev_specific_param = 0x80; /* Readonly.  */

    } else {

        dev_specific_param = 0x00;

    }



    if (req->cmd.buf[0] == MODE_SENSE) {

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



    bdrv_get_geometry(s->bs, &nb_sectors);

    if ((~dbd) & nb_sectors) {

        if (req->cmd.buf[0] == MODE_SENSE) {

            outbuf[3] = 8; /* Block descriptor length  */

        } else { /* MODE_SENSE_10 */

            outbuf[7] = 8; /* Block descriptor length  */

        }

        nb_sectors /= s->cluster_size;

        nb_sectors--;

        if (nb_sectors > 0xffffff)

            nb_sectors = 0xffffff;

        p[0] = 0; /* media density code */

        p[1] = (nb_sectors >> 16) & 0xff;

        p[2] = (nb_sectors >> 8) & 0xff;

        p[3] = nb_sectors & 0xff;

        p[4] = 0; /* reserved */

        p[5] = 0; /* bytes 5-7 are the sector size in bytes */

        p[6] = s->cluster_size * 2;

        p[7] = 0;

        p += 8;

    }



    switch (page) {

    case 0x04:

    case 0x05:

    case 0x08:

    case 0x2a:

        p += mode_sense_page(req, page, p);

        break;

    case 0x3f:

        p += mode_sense_page(req, 0x08, p);

        p += mode_sense_page(req, 0x2a, p);

        break;

    }



    buflen = p - outbuf;

    /*

     * The mode data length field specifies the length in bytes of the

     * following data that is available to be transferred. The mode data

     * length does not include itself.

     */

    if (req->cmd.buf[0] == MODE_SENSE) {

        outbuf[0] = buflen - 1;

    } else { /* MODE_SENSE_10 */

        outbuf[0] = ((buflen - 2) >> 8) & 0xff;

        outbuf[1] = (buflen - 2) & 0xff;

    }

    if (buflen > req->cmd.xfer)

        buflen = req->cmd.xfer;

    return buflen;

}
