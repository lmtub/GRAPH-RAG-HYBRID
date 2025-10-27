static void usb_msd_handle_data(USBDevice *dev, USBPacket *p)

{

    MSDState *s = (MSDState *)dev;

    uint32_t tag;

    struct usb_msd_cbw cbw;

    uint8_t devep = p->ep->nr;



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

            tag = le32_to_cpu(cbw.tag);

            s->data_len = le32_to_cpu(cbw.data_len);

            if (s->data_len == 0) {

                s->mode = USB_MSDM_CSW;

            } else if (cbw.flags & 0x80) {

                s->mode = USB_MSDM_DATAIN;

            } else {

                s->mode = USB_MSDM_DATAOUT;

            }

            DPRINTF("Command tag 0x%x flags %08x len %d data %d\n",

                    tag, cbw.flags, cbw.cmd_len, s->data_len);

            assert(le32_to_cpu(s->csw.residue) == 0);

            s->scsi_len = 0;

            s->req = scsi_req_new(s->scsi_dev, tag, 0, cbw.cmd, NULL);

#ifdef DEBUG_MSD

            scsi_req_print(s->req);

#endif

            scsi_req_enqueue(s->req);

            if (s->req && s->req->cmd.xfer != SCSI_XFER_NONE) {

                scsi_req_continue(s->req);

            }

            break;



        case USB_MSDM_DATAOUT:

            DPRINTF("Data out %zd/%d\n", p->iov.size, s->data_len);

            if (p->iov.size > s->data_len) {

                goto fail;

            }



            if (s->scsi_len) {

                usb_msd_copy_data(s, p);

            }

            if (le32_to_cpu(s->csw.residue)) {

                int len = p->iov.size - p->actual_length;

                if (len) {

                    usb_packet_skip(p, len);

                    s->data_len -= len;

                    if (s->data_len == 0) {

                        s->mode = USB_MSDM_CSW;

                    }

                }

            }

            if (p->actual_length < p->iov.size) {

                DPRINTF("Deferring packet %p [wait data-out]\n", p);

                s->packet = p;

                p->status = USB_RET_ASYNC;

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

            p->status = USB_RET_ASYNC;

            break;



        case USB_MSDM_CSW:

            if (p->iov.size < 13) {

                goto fail;

            }



            if (s->req) {

                /* still in flight */

                DPRINTF("Deferring packet %p [wait status]\n", p);

                s->packet = p;

                p->status = USB_RET_ASYNC;

            } else {

                usb_msd_send_status(s, p);

                s->mode = USB_MSDM_CBW;

            }

            break;



        case USB_MSDM_DATAIN:

            DPRINTF("Data in %zd/%d, scsi_len %d\n",

                    p->iov.size, s->data_len, s->scsi_len);

            if (s->scsi_len) {

                usb_msd_copy_data(s, p);

            }

            if (le32_to_cpu(s->csw.residue)) {

                int len = p->iov.size - p->actual_length;

                if (len) {

                    usb_packet_skip(p, len);

                    s->data_len -= len;

                    if (s->data_len == 0) {

                        s->mode = USB_MSDM_CSW;

                    }

                }

            }

            if (p->actual_length < p->iov.size) {

                DPRINTF("Deferring packet %p [wait data-in]\n", p);

                s->packet = p;

                p->status = USB_RET_ASYNC;

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

        p->status = USB_RET_STALL;

        break;

    }

}
