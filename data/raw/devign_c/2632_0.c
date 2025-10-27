int xen_config_dev_blk(DriveInfo *disk)

{

    char fe[256], be[256];

    int vdev = 202 * 256 + 16 * disk->unit;

    int cdrom = disk->bdrv->type == BDRV_TYPE_CDROM;

    const char *devtype = cdrom ? "cdrom" : "disk";

    const char *mode    = cdrom ? "r"     : "w";



    snprintf(disk->bdrv->device_name, sizeof(disk->bdrv->device_name),

	     "xvd%c", 'a' + disk->unit);

    xen_be_printf(NULL, 1, "config disk %d [%s]: %s\n",

                  disk->unit, disk->bdrv->device_name, disk->bdrv->filename);

    xen_config_dev_dirs("vbd", "qdisk", vdev, fe, be, sizeof(fe));



    /* frontend */

    xenstore_write_int(fe, "virtual-device",  vdev);

    xenstore_write_str(fe, "device-type",     devtype);



    /* backend */

    xenstore_write_str(be, "dev",             disk->bdrv->device_name);

    xenstore_write_str(be, "type",            "file");

    xenstore_write_str(be, "params",          disk->bdrv->filename);

    xenstore_write_str(be, "mode",            mode);



    /* common stuff */

    return xen_config_dev_all(fe, be);

}
