static int scsi_disk_emulate_inquiry(SCSIRequest *req, uint8_t *outbuf)

{

    SCSIDiskState *s = DO_UPCAST(SCSIDiskState, qdev, req->dev);

    int buflen = 0;



    if (req->cmd.buf[1] & 0x2) {

        /* Command support data - optional, not implemented */

        BADF("optional INQUIRY command support request not implemented\n");

        return -1;

    }



    if (req->cmd.buf[1] & 0x1) {

        /* Vital product data */

        uint8_t page_code = req->cmd.buf[2];

        if (req->cmd.xfer < 4) {

            BADF("Error: Inquiry (EVPD[%02X]) buffer size %zd is "

                 "less than 4\n", page_code, req->cmd.xfer);

            return -1;

        }



        if (bdrv_get_type_hint(s->bs) == BDRV_TYPE_CDROM) {

            outbuf[buflen++] = 5;

        } else {

            outbuf[buflen++] = 0;

        }

        outbuf[buflen++] = page_code ; // this page

        outbuf[buflen++] = 0x00;



        switch (page_code) {

        case 0x00: /* Supported page codes, mandatory */

            DPRINTF("Inquiry EVPD[Supported pages] "

                    "buffer size %zd\n", req->cmd.xfer);

            outbuf[buflen++] = 4;    // number of pages

            outbuf[buflen++] = 0x00; // list of supported pages (this page)

            outbuf[buflen++] = 0x80; // unit serial number

            outbuf[buflen++] = 0x83; // device identification

            outbuf[buflen++] = 0xb0; // block device characteristics

            break;



        case 0x80: /* Device serial number, optional */

        {

            const char *serial = req->dev->conf.dinfo->serial ?

                req->dev->conf.dinfo->serial : "0";

            int l = strlen(serial);



            if (l > req->cmd.xfer)

                l = req->cmd.xfer;

            if (l > 20)

                l = 20;



            DPRINTF("Inquiry EVPD[Serial number] "

                    "buffer size %zd\n", req->cmd.xfer);

            outbuf[buflen++] = l;

            memcpy(outbuf+buflen, serial, l);

            buflen += l;

            break;

        }



        case 0x83: /* Device identification page, mandatory */

        {

            int max_len = 255 - 8;

            int id_len = strlen(bdrv_get_device_name(s->bs));



            if (id_len > max_len)

                id_len = max_len;

            DPRINTF("Inquiry EVPD[Device identification] "

                    "buffer size %zd\n", req->cmd.xfer);



            outbuf[buflen++] = 3 + id_len;

            outbuf[buflen++] = 0x2; // ASCII

            outbuf[buflen++] = 0;   // not officially assigned

            outbuf[buflen++] = 0;   // reserved

            outbuf[buflen++] = id_len; // length of data following



            memcpy(outbuf+buflen, bdrv_get_device_name(s->bs), id_len);

            buflen += id_len;

            break;

        }

        case 0xb0: /* block device characteristics */

        {

            unsigned int min_io_size = s->qdev.conf.min_io_size >> 9;

            unsigned int opt_io_size = s->qdev.conf.opt_io_size >> 9;



            /* required VPD size with unmap support */

            outbuf[3] = buflen = 0x3c;



            memset(outbuf + 4, 0, buflen - 4);



            /* optimal transfer length granularity */

            outbuf[6] = (min_io_size >> 8) & 0xff;

            outbuf[7] = min_io_size & 0xff;



            /* optimal transfer length */

            outbuf[12] = (opt_io_size >> 24) & 0xff;

            outbuf[13] = (opt_io_size >> 16) & 0xff;

            outbuf[14] = (opt_io_size >> 8) & 0xff;

            outbuf[15] = opt_io_size & 0xff;

            break;

        }

        default:

            BADF("Error: unsupported Inquiry (EVPD[%02X]) "

                 "buffer size %zd\n", page_code, req->cmd.xfer);

            return -1;

        }

        /* done with EVPD */

        return buflen;

    }



    /* Standard INQUIRY data */

    if (req->cmd.buf[2] != 0) {

        BADF("Error: Inquiry (STANDARD) page or code "

             "is non-zero [%02X]\n", req->cmd.buf[2]);

        return -1;

    }



    /* PAGE CODE == 0 */

    if (req->cmd.xfer < 5) {

        BADF("Error: Inquiry (STANDARD) buffer size %zd "

             "is less than 5\n", req->cmd.xfer);

        return -1;

    }



    buflen = req->cmd.xfer;

    if (buflen > SCSI_MAX_INQUIRY_LEN)

        buflen = SCSI_MAX_INQUIRY_LEN;



    memset(outbuf, 0, buflen);



    if (req->lun || req->cmd.buf[1] >> 5) {

        outbuf[0] = 0x7f;	/* LUN not supported */

        return buflen;

    }



    if (bdrv_get_type_hint(s->bs) == BDRV_TYPE_CDROM) {

        outbuf[0] = 5;

        outbuf[1] = 0x80;

        memcpy(&outbuf[16], "QEMU CD-ROM     ", 16);

    } else {

        outbuf[0] = 0;

        memcpy(&outbuf[16], "QEMU HARDDISK   ", 16);

    }

    memcpy(&outbuf[8], "QEMU    ", 8);

    memcpy(&outbuf[32], s->version ? s->version : QEMU_VERSION, 4);

    /*

     * We claim conformance to SPC-3, which is required for guests

     * to ask for modern features like READ CAPACITY(16) or the

     * block characteristics VPD page by default.  Not all of SPC-3

     * is actually implemented, but we're good enough.

     */

    outbuf[2] = 5;

    outbuf[3] = 2; /* Format 2 */



    if (buflen > 36) {

        outbuf[4] = buflen - 5; /* Additional Length = (Len - 1) - 4 */

    } else {

        /* If the allocation length of CDB is too small,

               the additional length is not adjusted */

        outbuf[4] = 36 - 5;

    }



    /* Sync data transfer and TCQ.  */

    outbuf[7] = 0x10 | (req->bus->tcq ? 0x02 : 0);

    return buflen;

}
