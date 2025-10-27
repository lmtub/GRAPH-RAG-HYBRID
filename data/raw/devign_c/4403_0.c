static void async_complete(void *opaque)

{

    USBHostDevice *s = opaque;

    AsyncURB *aurb;

    int urbs = 0;



    while (1) {

        USBPacket *p;



        int r = ioctl(s->fd, USBDEVFS_REAPURBNDELAY, &aurb);

        if (r < 0) {

            if (errno == EAGAIN) {

                if (urbs > 2) {

                    fprintf(stderr, "husb: %d iso urbs finished at once\n", urbs);

                }

                return;

            }

            if (errno == ENODEV) {

                if (!s->closing) {

                    trace_usb_host_disconnect(s->bus_num, s->addr);

                    do_disconnect(s);

                }

                return;

            }



            perror("USBDEVFS_REAPURBNDELAY");

            return;

        }



        DPRINTF("husb: async completed. aurb %p status %d alen %d\n",

                aurb, aurb->urb.status, aurb->urb.actual_length);



        /* If this is a buffered iso urb mark it as complete and don't do

           anything else (it is handled further in usb_host_handle_iso_data) */

        if (aurb->iso_frame_idx == -1) {

            int inflight;

            int pid = (aurb->urb.endpoint & USB_DIR_IN) ?

                USB_TOKEN_IN : USB_TOKEN_OUT;

            int ep = aurb->urb.endpoint & 0xf;

            if (aurb->urb.status == -EPIPE) {

                set_halt(s, pid, ep);

            }

            aurb->iso_frame_idx = 0;

            urbs++;

            inflight = change_iso_inflight(s, pid, ep, -1);

            if (inflight == 0 && is_iso_started(s, pid, ep)) {

                fprintf(stderr, "husb: out of buffers for iso stream\n");

            }

            continue;

        }



        p = aurb->packet;

        trace_usb_host_urb_complete(s->bus_num, s->addr, aurb, aurb->urb.status,

                                    aurb->urb.actual_length, aurb->more);



        if (p) {

            switch (aurb->urb.status) {

            case 0:

                p->result += aurb->urb.actual_length;

                break;



            case -EPIPE:

                set_halt(s, p->pid, p->devep);

                p->result = USB_RET_STALL;

                break;



            default:

                p->result = USB_RET_NAK;

                break;

            }



            if (aurb->urb.type == USBDEVFS_URB_TYPE_CONTROL) {

                trace_usb_host_req_complete(s->bus_num, s->addr, p->result);

                usb_generic_async_ctrl_complete(&s->dev, p);

            } else if (!aurb->more) {

                trace_usb_host_req_complete(s->bus_num, s->addr, p->result);

                usb_packet_complete(&s->dev, p);

            }

        }



        async_free(aurb);

    }

}
