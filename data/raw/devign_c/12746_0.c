static int scsi_block_initfn(SCSIDevice *dev)

{

    SCSIDiskState *s = DO_UPCAST(SCSIDiskState, qdev, dev);

    int sg_version;

    int rc;



    if (!s->qdev.conf.bs) {

        error_report("drive property not set");

        return -1;

    }



    /* check we are using a driver managing SG_IO (version 3 and after) */

    rc = bdrv_ioctl(s->qdev.conf.bs, SG_GET_VERSION_NUM, &sg_version);

    if (rc < 0) {

        error_report("cannot get SG_IO version number: %s.  "

                     "Is this a SCSI device?",

                     strerror(-rc));

        return -1;

    }

    if (sg_version < 30000) {

        error_report("scsi generic interface too old");

        return -1;

    }



    /* get device type from INQUIRY data */

    rc = get_device_type(s);

    if (rc < 0) {

        error_report("INQUIRY failed");

        return -1;

    }



    /* Make a guess for the block size, we'll fix it when the guest sends.

     * READ CAPACITY.  If they don't, they likely would assume these sizes

     * anyway. (TODO: check in /sys).

     */

    if (s->qdev.type == TYPE_ROM || s->qdev.type == TYPE_WORM) {

        s->qdev.blocksize = 2048;

    } else {

        s->qdev.blocksize = 512;

    }



    /* Makes the scsi-block device not removable by using HMP and QMP eject

     * command.

     */

    s->features |= (1 << SCSI_DISK_F_NO_REMOVABLE_DEVOPS);



    return scsi_initfn(&s->qdev);

}
