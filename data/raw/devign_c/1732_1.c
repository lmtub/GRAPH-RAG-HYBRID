static void usb_msd_command_complete(SCSIBus *bus, int reason, uint32_t tag,

                                     uint32_t arg)

{

    MSDState *s = DO_UPCAST(MSDState, dev.qdev, bus->qbus.parent);

    USBPacket *p = s->packet;



    if (tag != s->tag) {

        fprintf(stderr, "usb-msd: Unexpected SCSI Tag 0x%x\n", tag);

    }

    if (reason == SCSI_REASON_DONE) {

        DPRINTF("Command complete %d\n", arg);

        s->residue = s->data_len;

        s->result = arg != 0;

        if (s->packet) {

            if (s->data_len == 0 && s->mode == USB_MSDM_DATAOUT) {

                /* A deferred packet with no write data remaining must be

                   the status read packet.  */

                usb_msd_send_status(s, p);

                s->mode = USB_MSDM_CBW;

            } else {

                if (s->data_len) {

                    s->data_len -= s->usb_len;

                    if (s->mode == USB_MSDM_DATAIN)

                        memset(s->usb_buf, 0, s->usb_len);

                    s->usb_len = 0;

                }

                if (s->data_len == 0)

                    s->mode = USB_MSDM_CSW;

            }

            s->packet = NULL;

            usb_packet_complete(&s->dev, p);

        } else if (s->data_len == 0) {

            s->mode = USB_MSDM_CSW;

        }

        return;

    }

    s->scsi_len = arg;

    s->scsi_buf = s->scsi_dev->info->get_buf(s->scsi_dev, tag);

    if (p) {

        usb_msd_copy_data(s);

        if (s->usb_len == 0) {

            /* Set s->packet to NULL before calling usb_packet_complete

               because another request may be issued before

               usb_packet_complete returns.  */

            DPRINTF("Packet complete %p\n", p);

            s->packet = NULL;

            usb_packet_complete(&s->dev, p);

        }

    }

}
