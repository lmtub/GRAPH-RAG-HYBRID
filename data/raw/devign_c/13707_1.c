static void scsi_disk_set_sense(SCSIDiskState *s, uint8_t key)

{

    s->sense.key = key;

}
