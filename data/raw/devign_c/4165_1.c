static void usb_msd_cancel_io(USBDevice *dev, USBPacket *p)

{

    MSDState *s = DO_UPCAST(MSDState, dev, dev);

    scsi_req_cancel(s->req);

}
