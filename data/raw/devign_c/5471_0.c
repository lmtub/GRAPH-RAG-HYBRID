static void scsi_destroy(SCSIDevice *s)

{

    scsi_device_purge_requests(s, SENSE_CODE(NO_SENSE));

    blockdev_mark_auto_del(s->conf.bs);

}
