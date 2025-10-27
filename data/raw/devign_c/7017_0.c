static int scsi_device_init(SCSIDevice *s)

{

    SCSIDeviceClass *sc = SCSI_DEVICE_GET_CLASS(s);

    if (sc->init) {

        return sc->init(s);

    }

    return 0;

}
