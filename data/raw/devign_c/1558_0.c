static int usb_host_handle_iso_data(USBHostDevice *s, USBPacket *p, int in)

{

    AsyncURB *aurb;

    int i, j, ret, max_packet_size, offset, len = 0;

    uint8_t *buf;



    max_packet_size = usb_ep_get_max_packet_size(&s->dev, p->pid, p->devep);

    if (max_packet_size == 0)

        return USB_RET_NAK;



    aurb = get_iso_urb(s, p->pid, p->devep);

    if (!aurb) {

        aurb = usb_host_alloc_iso(s, p->pid, p->devep);

    }



    i = get_iso_urb_idx(s, p->pid, p->devep);

    j = aurb[i].iso_frame_idx;

    if (j >= 0 && j < ISO_FRAME_DESC_PER_URB) {

        if (in) {

            /* Check urb status  */

            if (aurb[i].urb.status) {

                len = urb_status_to_usb_ret(aurb[i].urb.status);

                /* Move to the next urb */

                aurb[i].iso_frame_idx = ISO_FRAME_DESC_PER_URB - 1;

            /* Check frame status */

            } else if (aurb[i].urb.iso_frame_desc[j].status) {

                len = urb_status_to_usb_ret(

                                        aurb[i].urb.iso_frame_desc[j].status);

            /* Check the frame fits */

            } else if (aurb[i].urb.iso_frame_desc[j].actual_length

                       > p->iov.size) {

                printf("husb: received iso data is larger then packet\n");

                len = USB_RET_NAK;

            /* All good copy data over */

            } else {

                len = aurb[i].urb.iso_frame_desc[j].actual_length;

                buf  = aurb[i].urb.buffer +

                    j * aurb[i].urb.iso_frame_desc[0].length;

                usb_packet_copy(p, buf, len);

            }

        } else {

            len = p->iov.size;

            offset = (j == 0) ? 0 : get_iso_buffer_used(s, p->pid, p->devep);



            /* Check the frame fits */

            if (len > max_packet_size) {

                printf("husb: send iso data is larger then max packet size\n");

                return USB_RET_NAK;

            }



            /* All good copy data over */

            usb_packet_copy(p, aurb[i].urb.buffer + offset, len);

            aurb[i].urb.iso_frame_desc[j].length = len;

            offset += len;

            set_iso_buffer_used(s, p->pid, p->devep, offset);



            /* Start the stream once we have buffered enough data */

            if (!is_iso_started(s, p->pid, p->devep) && i == 1 && j == 8) {

                set_iso_started(s, p->pid, p->devep);

            }

        }

        aurb[i].iso_frame_idx++;

        if (aurb[i].iso_frame_idx == ISO_FRAME_DESC_PER_URB) {

            i = (i + 1) % s->iso_urb_count;

            set_iso_urb_idx(s, p->pid, p->devep, i);

        }

    } else {

        if (in) {

            set_iso_started(s, p->pid, p->devep);

        } else {

            DPRINTF("hubs: iso out error no free buffer, dropping packet\n");

        }

    }



    if (is_iso_started(s, p->pid, p->devep)) {

        /* (Re)-submit all fully consumed / filled urbs */

        for (i = 0; i < s->iso_urb_count; i++) {

            if (aurb[i].iso_frame_idx == ISO_FRAME_DESC_PER_URB) {

                ret = ioctl(s->fd, USBDEVFS_SUBMITURB, &aurb[i]);

                if (ret < 0) {

                    perror("USBDEVFS_SUBMITURB");

                    if (!in || len == 0) {

                        switch(errno) {

                        case ETIMEDOUT:

                            len = USB_RET_NAK;

                            break;

                        case EPIPE:

                        default:

                            len = USB_RET_STALL;

                        }

                    }

                    break;

                }

                aurb[i].iso_frame_idx = -1;

                change_iso_inflight(s, p->pid, p->devep, 1);

            }

        }

    }



    return len;

}
