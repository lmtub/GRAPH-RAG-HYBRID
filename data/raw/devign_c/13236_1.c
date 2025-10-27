static void ohci_async_cancel_device(OHCIState *ohci, USBDevice *dev)

{

    if (ohci->async_td &&

        ohci->usb_packet.owner != NULL &&

        ohci->usb_packet.owner->dev == dev) {

        usb_cancel_packet(&ohci->usb_packet);

        ohci->async_td = 0;

    }

}
