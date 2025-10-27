static int scsi_generic_initfn(SCSIDevice *s)

{

    int rc;

    int sg_version;

    struct sg_scsi_id scsiid;



    if (!s->conf.bs) {

        error_report("drive property not set");

        return -1;

    }



    if (bdrv_get_on_error(s->conf.bs, 0) != BLOCKDEV_ON_ERROR_ENOSPC) {

        error_report("Device doesn't support drive option werror");

        return -1;

    }

    if (bdrv_get_on_error(s->conf.bs, 1) != BLOCKDEV_ON_ERROR_REPORT) {

        error_report("Device doesn't support drive option rerror");

        return -1;

    }



    /* check we are using a driver managing SG_IO (version 3 and after */

    rc = bdrv_ioctl(s->conf.bs, SG_GET_VERSION_NUM, &sg_version);

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



    /* get LUN of the /dev/sg? */

    if (bdrv_ioctl(s->conf.bs, SG_GET_SCSI_ID, &scsiid)) {

        error_report("SG_GET_SCSI_ID ioctl failed");

        return -1;

    }



    /* define device state */

    s->type = scsiid.scsi_type;

    DPRINTF("device type %d\n", s->type);

    if (s->type == TYPE_DISK || s->type == TYPE_ROM) {

        add_boot_device_path(s->conf.bootindex, &s->qdev, NULL);

    }



    switch (s->type) {

    case TYPE_TAPE:

        s->blocksize = get_stream_blocksize(s->conf.bs);

        if (s->blocksize == -1) {

            s->blocksize = 0;

        }

        break;



        /* Make a guess for block devices, we'll fix it when the guest sends.

         * READ CAPACITY.  If they don't, they likely would assume these sizes

         * anyway. (TODO: they could also send MODE SENSE).

         */

    case TYPE_ROM:

    case TYPE_WORM:

        s->blocksize = 2048;

        break;

    default:

        s->blocksize = 512;

        break;

    }



    DPRINTF("block size %d\n", s->blocksize);

    return 0;

}
