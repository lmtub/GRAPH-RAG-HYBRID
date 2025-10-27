static int scsi_cd_initfn(SCSIDevice *dev)

{

    SCSIDiskState *s = DO_UPCAST(SCSIDiskState, qdev, dev);

    s->qdev.blocksize = 2048;

    s->qdev.type = TYPE_ROM;

    s->features |= 1 << SCSI_DISK_F_REMOVABLE;

    if (!s->product) {

        s->product = g_strdup("QEMU CD-ROM");

    }

    return scsi_initfn(&s->qdev);

}
