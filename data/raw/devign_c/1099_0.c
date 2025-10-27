static void scsi_unrealize(SCSIDevice *s, Error **errp)

{

    scsi_device_purge_requests(s, SENSE_CODE(NO_SENSE));

    blockdev_mark_auto_del(s->conf.blk);

}
