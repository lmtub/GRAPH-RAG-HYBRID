static void scsi_disk_class_initfn(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    SCSIDeviceClass *sc = SCSI_DEVICE_CLASS(klass);



    sc->init         = scsi_disk_initfn;

    sc->destroy      = scsi_destroy;

    sc->alloc_req    = scsi_new_request;

    sc->unit_attention_reported = scsi_disk_unit_attention_reported;

    dc->fw_name = "disk";

    dc->desc = "virtual SCSI disk or CD-ROM (legacy)";

    dc->reset = scsi_disk_reset;

    dc->props = scsi_disk_properties;

    dc->vmsd  = &vmstate_scsi_disk_state;

}
