static void musb_async_cancel_device(MUSBState *s, USBDevice *dev)

{

    int ep, dir;



    for (ep = 0; ep < 16; ep++) {

        for (dir = 0; dir < 2; dir++) {

            if (s->ep[ep].packey[dir].p.owner == NULL ||

                s->ep[ep].packey[dir].p.owner->dev != dev) {

                continue;

            }

            usb_cancel_packet(&s->ep[ep].packey[dir].p);

            /* status updates needed here? */

        }

    }

}
