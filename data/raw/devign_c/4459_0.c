static int usb_msd_handle_data(USBDevice *dev, USBPacket *p)

{

    MSDState *s = (MSDState *)dev;

    int ret = 0;

    struct usb_msd_cbw cbw;

    uint8_t devep = p->devep;



    switch (p->pid) {

    case USB_TOKEN_OUT:

        if (devep != 2)

            goto fail;



        switch (s->mode) {

        case USB_MSDM_CBW:

            if (p->iov.size != 31) {

                fprintf(stderr, "usb-msd: Bad CBW size");

                goto fail;

            }

            usb_packet_copy(p, &cbw, 31);

            if (le32_to_cpu(cbw.sig) != 0x43425355) {

                fprintf(stderr, "usb-msd: Bad signature %08x\n",

                        le32_to_cpu(cbw.sig));

                goto fail;

            }

            DPRINTF("Command on LUN %d\n", cbw.lun);

            if (cbw.lun != 0) {

                fprintf(stderr, "usb-msd: Bad LUN %d\n", cbw.lun);

                goto fail;

            }

            s->tag = le32_to_cpu(cbw.tag);

            s->data_len = le32_to_cpu(cbw.data_len);

            if (s->data_len == 0) {

                s->mode = USB_MSDM_CSW;

            } else if (cbw.flags & 0x80) {

                s->mode = USB_MSDM_DATAIN;

            } else {

                s->mode = USB_MSDM_DATAOUT;

            }

            DPRINTF("Command tag 0x%x flags %08x len %d data %d\n",

                    s->tag, cbw.flags, cbw.cmd_len, s->data_len);

            s->residue = 0;

            s->scsi_len = 0;

            s->req = scsi_req_new(s->scsi_dev, s->tag, 0, NULL);

            scsi_req_enqueue(s->req, cbw.cmd);

            /* ??? Should check that USB and SCSI data transfer

               directions match.  */

            if (s->mode != USB_MSDM_CSW && s->residue == 0) {

                scsi_req_continue(s->req);

            }

            ret = p->result;

            break;



        case USB_MSDM_DATAOUT:

            DPRINTF("Data out %zd/%d\n", p->iov.size, s->data_len);

            if (p->iov.size > s->data_len) {

                goto fail;

            }



            if (s->scsi_len) {

                usb_msd_copy_data(s, p);

            }

            if (s->residue) {

                int len = p->iov.size - p->result;

                if (len) {

                    usb_packet_skip(p, len);

                    s->data_len -= len;

                    if (s->data_len == 0) {

                        s->mode = USB_MSDM_CSW;

                    }

                }

            }

            if (p->result < p->iov.size) {

                DPRINTF("Deferring packet %p\n", p);

                s->packet = p;

                ret = USB_RET_ASYNC;

            } else {

                ret = p->result;

            }

            break;



        default:

            DPRINTF("Unexpected write (len %zd)\n", p->iov.size);

            goto fail;

        }

        break;



    case USB_TOKEN_IN:

        if (devep != 1)

            goto fail;



        switch (s->mode) {

        case USB_MSDM_DATAOUT:

            if (s->data_len != 0 || p->iov.size < 13) {

                goto fail;

            }

            /* Waiting for SCSI write to complete.  */

            s->packet = p;

            ret = USB_RET_ASYNC;

            break;



        case USB_MSDM_CSW:

            DPRINTF("Command status %d tag 0x%x, len %zd\n",

                    s->result, s->tag, p->iov.size);

            if (p->iov.size < 13) {

                goto fail;

            }



            usb_msd_send_status(s, p);

            s->mode = USB_MSDM_CBW;

            ret = 13;

            break;



        case USB_MSDM_DATAIN:

            DPRINTF("Data in %zd/%d, scsi_len %d\n",

                    p->iov.size, s->data_len, s->scsi_len);

            if (s->scsi_len) {

                usb_msd_copy_data(s, p);

            }

            if (s->residue) {

                int len = p->iov.size - p->result;

                if (len) {

                    usb_packet_skip(p, len);

                    s->data_len -= len;

                    if (s->data_len == 0) {

                        s->mode = USB_MSDM_CSW;

                    }

                }

            }

            if (p->result < p->iov.size) {

                DPRINTF("Deferring packet %p\n", p);

                s->packet = p;

                ret = USB_RET_ASYNC;

            } else {

                ret = p->result;

            }

            break;



        default:

            DPRINTF("Unexpected read (len %zd)\n", p->iov.size);

            goto fail;

        }

        break;



    default:

        DPRINTF("Bad token\n");

    fail:

        ret = USB_RET_STALL;

        break;

    }



    return ret;

}
