static void lsi_do_msgout(LSIState *s)

{

    uint8_t msg;

    int len;

    uint32_t current_tag;

    SCSIDevice *current_dev;

    lsi_request *p, *p_next;

    int id;



    if (s->current) {

        current_tag = s->current->tag;

    } else {

        current_tag = s->select_tag;

    }

    id = (current_tag >> 8) & 0xf;

    current_dev = s->bus.devs[id];



    DPRINTF("MSG out len=%d\n", s->dbc);

    while (s->dbc) {

        msg = lsi_get_msgbyte(s);

        s->sfbr = msg;



        switch (msg) {

        case 0x04:

            DPRINTF("MSG: Disconnect\n");

            lsi_disconnect(s);

            break;

        case 0x08:

            DPRINTF("MSG: No Operation\n");

            lsi_set_phase(s, PHASE_CMD);

            break;

        case 0x01:

            len = lsi_get_msgbyte(s);

            msg = lsi_get_msgbyte(s);

            (void)len; /* avoid a warning about unused variable*/

            DPRINTF("Extended message 0x%x (len %d)\n", msg, len);

            switch (msg) {

            case 1:

                DPRINTF("SDTR (ignored)\n");

                lsi_skip_msgbytes(s, 2);

                break;

            case 3:

                DPRINTF("WDTR (ignored)\n");

                lsi_skip_msgbytes(s, 1);

                break;

            default:

                goto bad;

            }

            break;

        case 0x20: /* SIMPLE queue */

            s->select_tag |= lsi_get_msgbyte(s) | LSI_TAG_VALID;

            DPRINTF("SIMPLE queue tag=0x%x\n", s->select_tag & 0xff);

            break;

        case 0x21: /* HEAD of queue */

            BADF("HEAD queue not implemented\n");

            s->select_tag |= lsi_get_msgbyte(s) | LSI_TAG_VALID;

            break;

        case 0x22: /* ORDERED queue */

            BADF("ORDERED queue not implemented\n");

            s->select_tag |= lsi_get_msgbyte(s) | LSI_TAG_VALID;

            break;

        case 0x0d:

            /* The ABORT TAG message clears the current I/O process only. */

            DPRINTF("MSG: ABORT TAG tag=0x%x\n", current_tag);

            current_dev->info->cancel_io(current_dev, current_tag);

            lsi_disconnect(s);

            break;

        case 0x06:

        case 0x0e:

        case 0x0c:

            /* The ABORT message clears all I/O processes for the selecting

               initiator on the specified logical unit of the target. */

            if (msg == 0x06) {

                DPRINTF("MSG: ABORT tag=0x%x\n", current_tag);

            }

            /* The CLEAR QUEUE message clears all I/O processes for all

               initiators on the specified logical unit of the target. */

            if (msg == 0x0e) {

                DPRINTF("MSG: CLEAR QUEUE tag=0x%x\n", current_tag);

            }

            /* The BUS DEVICE RESET message clears all I/O processes for all

               initiators on all logical units of the target. */

            if (msg == 0x0c) {

                DPRINTF("MSG: BUS DEVICE RESET tag=0x%x\n", current_tag);

            }



            /* clear the current I/O process */

            current_dev->info->cancel_io(current_dev, current_tag);



            /* As the current implemented devices scsi_disk and scsi_generic

               only support one LUN, we don't need to keep track of LUNs.

               Clearing I/O processes for other initiators could be possible

               for scsi_generic by sending a SG_SCSI_RESET to the /dev/sgX

               device, but this is currently not implemented (and seems not

               to be really necessary). So let's simply clear all queued

               commands for the current device: */

            id = current_tag & 0x0000ff00;

            QTAILQ_FOREACH_SAFE(p, &s->queue, next, p_next) {

                if ((p->tag & 0x0000ff00) == id) {

                    current_dev->info->cancel_io(current_dev, p->tag);

                    QTAILQ_REMOVE(&s->queue, p, next);

                }

            }



            lsi_disconnect(s);

            break;

        default:

            if ((msg & 0x80) == 0) {

                goto bad;

            }

            s->current_lun = msg & 7;

            DPRINTF("Select LUN %d\n", s->current_lun);

            lsi_set_phase(s, PHASE_CMD);

            break;

        }

    }

    return;

bad:

    BADF("Unimplemented message 0x%02x\n", msg);

    lsi_set_phase(s, PHASE_MI);

    lsi_add_msg_byte(s, 7); /* MESSAGE REJECT */

    s->msg_action = 0;

}
