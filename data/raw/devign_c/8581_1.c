static int scsi_disk_emulate_inquiry(SCSIRequest *req, uint8_t *outbuf)

{

    SCSIDiskState *s = DO_UPCAST(SCSIDiskState, qdev, req->dev);

    int buflen = 0;

    int start;



    if (req->cmd.buf[1] & 0x1) {

        /* Vital product data */

        uint8_t page_code = req->cmd.buf[2];



        outbuf[buflen++] = s->qdev.type & 0x1f;

        outbuf[buflen++] = page_code ; // this page

        outbuf[buflen++] = 0x00;

        outbuf[buflen++] = 0x00;

        start = buflen;



        switch (page_code) {

        case 0x00: /* Supported page codes, mandatory */

        {

            DPRINTF("Inquiry EVPD[Supported pages] "

                    "buffer size %zd\n", req->cmd.xfer);

            outbuf[buflen++] = 0x00; // list of supported pages (this page)

            if (s->serial) {

                outbuf[buflen++] = 0x80; // unit serial number

            }

            outbuf[buflen++] = 0x83; // device identification

            if (s->qdev.type == TYPE_DISK) {

                outbuf[buflen++] = 0xb0; // block limits

                outbuf[buflen++] = 0xb2; // thin provisioning

            }

            break;

        }

        case 0x80: /* Device serial number, optional */

        {

            int l;



            if (!s->serial) {

                DPRINTF("Inquiry (EVPD[Serial number] not supported\n");

                return -1;

            }



            l = strlen(s->serial);

            if (l > 20) {

                l = 20;

            }



            DPRINTF("Inquiry EVPD[Serial number] "

                    "buffer size %zd\n", req->cmd.xfer);

            memcpy(outbuf+buflen, s->serial, l);

            buflen += l;

            break;

        }



        case 0x83: /* Device identification page, mandatory */

        {

            const char *str = s->serial ?: bdrv_get_device_name(s->qdev.conf.bs);

            int max_len = s->serial ? 20 : 255 - 8;

            int id_len = strlen(str);



            if (id_len > max_len) {

                id_len = max_len;

            }

            DPRINTF("Inquiry EVPD[Device identification] "

                    "buffer size %zd\n", req->cmd.xfer);



            outbuf[buflen++] = 0x2; // ASCII

            outbuf[buflen++] = 0;   // not officially assigned

            outbuf[buflen++] = 0;   // reserved

            outbuf[buflen++] = id_len; // length of data following

            memcpy(outbuf+buflen, str, id_len);

            buflen += id_len;



            if (s->wwn) {

                outbuf[buflen++] = 0x1; // Binary

                outbuf[buflen++] = 0x3; // NAA

                outbuf[buflen++] = 0;   // reserved

                outbuf[buflen++] = 8;

                stq_be_p(&outbuf[buflen], s->wwn);

                buflen += 8;

            }

            break;

        }

        case 0xb0: /* block limits */

        {

            unsigned int unmap_sectors =

                    s->qdev.conf.discard_granularity / s->qdev.blocksize;

            unsigned int min_io_size =

                    s->qdev.conf.min_io_size / s->qdev.blocksize;

            unsigned int opt_io_size =

                    s->qdev.conf.opt_io_size / s->qdev.blocksize;



            if (s->qdev.type == TYPE_ROM) {

                DPRINTF("Inquiry (EVPD[%02X] not supported for CDROM\n",

                        page_code);

                return -1;

            }

            /* required VPD size with unmap support */

            buflen = 0x40;

            memset(outbuf + 4, 0, buflen - 4);



            /* optimal transfer length granularity */

            outbuf[6] = (min_io_size >> 8) & 0xff;

            outbuf[7] = min_io_size & 0xff;



            /* optimal transfer length */

            outbuf[12] = (opt_io_size >> 24) & 0xff;

            outbuf[13] = (opt_io_size >> 16) & 0xff;

            outbuf[14] = (opt_io_size >> 8) & 0xff;

            outbuf[15] = opt_io_size & 0xff;



            /* optimal unmap granularity */

            outbuf[28] = (unmap_sectors >> 24) & 0xff;

            outbuf[29] = (unmap_sectors >> 16) & 0xff;

            outbuf[30] = (unmap_sectors >> 8) & 0xff;

            outbuf[31] = unmap_sectors & 0xff;

            break;

        }

        case 0xb2: /* thin provisioning */

        {

            buflen = 8;

            outbuf[4] = 0;

            outbuf[5] = 0xe0; /* unmap & write_same 10/16 all supported */

            outbuf[6] = s->qdev.conf.discard_granularity ? 2 : 1;

            outbuf[7] = 0;

            break;

        }

        default:

            return -1;

        }

        /* done with EVPD */

        assert(buflen - start <= 255);

        outbuf[start - 1] = buflen - start;

        return buflen;

    }



    /* Standard INQUIRY data */

    if (req->cmd.buf[2] != 0) {

        return -1;

    }



    /* PAGE CODE == 0 */

    buflen = req->cmd.xfer;

    if (buflen > SCSI_MAX_INQUIRY_LEN) {

        buflen = SCSI_MAX_INQUIRY_LEN;

    }

    memset(outbuf, 0, buflen);



    outbuf[0] = s->qdev.type & 0x1f;

    outbuf[1] = (s->features & (1 << SCSI_DISK_F_REMOVABLE)) ? 0x80 : 0;



    strpadcpy((char *) &outbuf[16], 16, s->product, ' ');

    strpadcpy((char *) &outbuf[8], 8, s->vendor, ' ');



    memset(&outbuf[32], 0, 4);

    memcpy(&outbuf[32], s->version, MIN(4, strlen(s->version)));

    /*

     * We claim conformance to SPC-3, which is required for guests

     * to ask for modern features like READ CAPACITY(16) or the

     * block characteristics VPD page by default.  Not all of SPC-3

     * is actually implemented, but we're good enough.

     */

    outbuf[2] = 5;

    outbuf[3] = 2 | 0x10; /* Format 2, HiSup */



    if (buflen > 36) {

        outbuf[4] = buflen - 5; /* Additional Length = (Len - 1) - 4 */

    } else {

        /* If the allocation length of CDB is too small,

               the additional length is not adjusted */

        outbuf[4] = 36 - 5;

    }



    /* Sync data transfer and TCQ.  */

    outbuf[7] = 0x10 | (req->bus->info->tcq ? 0x02 : 0);

    return buflen;

}
