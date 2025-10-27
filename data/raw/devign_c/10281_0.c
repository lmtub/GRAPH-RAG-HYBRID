static void scsi_device_unrealize(SCSIDevice *s, Error **errp)

{

    SCSIDeviceClass *sc = SCSI_DEVICE_GET_CLASS(s);

    if (sc->unrealize) {

        sc->unrealize(s, errp);

    }

}
