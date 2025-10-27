void usb_ep_reset(USBDevice *dev)

{

    int ep;



    dev->ep_ctl.nr = 0;

    dev->ep_ctl.type = USB_ENDPOINT_XFER_CONTROL;

    dev->ep_ctl.ifnum = 0;

    dev->ep_ctl.dev = dev;

    dev->ep_ctl.pipeline = false;

    for (ep = 0; ep < USB_MAX_ENDPOINTS; ep++) {

        dev->ep_in[ep].nr = ep + 1;

        dev->ep_out[ep].nr = ep + 1;

        dev->ep_in[ep].pid = USB_TOKEN_IN;

        dev->ep_out[ep].pid = USB_TOKEN_OUT;

        dev->ep_in[ep].type = USB_ENDPOINT_XFER_INVALID;

        dev->ep_out[ep].type = USB_ENDPOINT_XFER_INVALID;

        dev->ep_in[ep].ifnum = 0;

        dev->ep_out[ep].ifnum = 0;

        dev->ep_in[ep].dev = dev;

        dev->ep_out[ep].dev = dev;

        dev->ep_in[ep].pipeline = false;

        dev->ep_out[ep].pipeline = false;

    }

}
