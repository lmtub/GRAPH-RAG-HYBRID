void usb_ep_combine_input_packets(USBEndpoint *ep)

{

    USBPacket *p, *u, *next, *prev = NULL, *first = NULL;

    USBPort *port = ep->dev->port;

    int ret;



    assert(ep->pipeline);

    assert(ep->pid == USB_TOKEN_IN);



    QTAILQ_FOREACH_SAFE(p, &ep->queue, queue, next) {

        /* Empty the queue on a halt */

        if (ep->halted) {

            p->result = USB_RET_REMOVE_FROM_QUEUE;

            port->ops->complete(port, p);

            continue;

        }



        /* Skip packets already submitted to the device */

        if (p->state == USB_PACKET_ASYNC) {

            prev = p;

            continue;

        }

        usb_packet_check_state(p, USB_PACKET_QUEUED);



        /*

         * If the previous (combined) packet has the short_not_ok flag set

         * stop, as we must not submit packets to the device after a transfer

         * ending with short_not_ok packet.

         */

        if (prev && prev->short_not_ok) {

            break;

        }



        if (first) {

            if (first->combined == NULL) {

                USBCombinedPacket *combined = g_new0(USBCombinedPacket, 1);



                combined->first = first;

                QTAILQ_INIT(&combined->packets);

                qemu_iovec_init(&combined->iov, 2);

                usb_combined_packet_add(combined, first);

            }

            usb_combined_packet_add(first->combined, p);

        } else {

            first = p;

        }



        /* Is this packet the last one of a (combined) transfer? */

        if ((p->iov.size % ep->max_packet_size) != 0 || !p->short_not_ok ||

                next == NULL) {

            ret = usb_device_handle_data(ep->dev, first);

            assert(ret == USB_RET_ASYNC);

            if (first->combined) {

                QTAILQ_FOREACH(u, &first->combined->packets, combined_entry) {

                    usb_packet_set_state(u, USB_PACKET_ASYNC);

                }

            } else {

                usb_packet_set_state(first, USB_PACKET_ASYNC);

            }

            first = NULL;

            prev = p;

        }

    }

}
