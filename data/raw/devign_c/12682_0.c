static int blk_init(struct XenDevice *xendev)

{

    struct XenBlkDev *blkdev = container_of(xendev, struct XenBlkDev, xendev);

    int index, qflags, have_barriers, info = 0;

    char *h;



    /* read xenstore entries */

    if (blkdev->params == NULL) {

	blkdev->params = xenstore_read_be_str(&blkdev->xendev, "params");

        h = strchr(blkdev->params, ':');

	if (h != NULL) {

	    blkdev->fileproto = blkdev->params;

	    blkdev->filename  = h+1;

	    *h = 0;

	} else {

	    blkdev->fileproto = "<unset>";

	    blkdev->filename  = blkdev->params;

	}

    }

    if (blkdev->mode == NULL)

	blkdev->mode = xenstore_read_be_str(&blkdev->xendev, "mode");

    if (blkdev->type == NULL)

	blkdev->type = xenstore_read_be_str(&blkdev->xendev, "type");

    if (blkdev->dev == NULL)

	blkdev->dev = xenstore_read_be_str(&blkdev->xendev, "dev");

    if (blkdev->devtype == NULL)

	blkdev->devtype = xenstore_read_be_str(&blkdev->xendev, "device-type");



    /* do we have all we need? */

    if (blkdev->params == NULL ||

	blkdev->mode == NULL   ||

	blkdev->type == NULL   ||

	blkdev->dev == NULL)

	return -1;



    /* read-only ? */

    if (strcmp(blkdev->mode, "w") == 0) {

	qflags = BDRV_O_RDWR;

    } else {

	qflags = 0;

	info  |= VDISK_READONLY;

    }



    /* cdrom ? */

    if (blkdev->devtype && !strcmp(blkdev->devtype, "cdrom"))

	info  |= VDISK_CDROM;



    /* init qemu block driver */

    index = (blkdev->xendev.dev - 202 * 256) / 16;

    blkdev->dinfo = drive_get(IF_XEN, 0, index);

    if (!blkdev->dinfo) {

        /* setup via xenbus -> create new block driver instance */

        xen_be_printf(&blkdev->xendev, 2, "create new bdrv (xenbus setup)\n");

	blkdev->bs = bdrv_new(blkdev->dev);

	if (blkdev->bs) {

	    if (bdrv_open(blkdev->bs, blkdev->filename, qflags,

                           bdrv_find_whitelisted_format(blkdev->fileproto))

                != 0) {

		bdrv_delete(blkdev->bs);

		blkdev->bs = NULL;

	    }

	}

	if (!blkdev->bs)

	    return -1;

    } else {

        /* setup via qemu cmdline -> already setup for us */

        xen_be_printf(&blkdev->xendev, 2, "get configured bdrv (cmdline setup)\n");

	blkdev->bs = blkdev->dinfo->bdrv;

    }

    blkdev->file_blk  = BLOCK_SIZE;

    blkdev->file_size = bdrv_getlength(blkdev->bs);

    if (blkdev->file_size < 0) {

        xen_be_printf(&blkdev->xendev, 1, "bdrv_getlength: %d (%s) | drv %s\n",

                      (int)blkdev->file_size, strerror(-blkdev->file_size),

                      blkdev->bs->drv ? blkdev->bs->drv->format_name : "-");

	blkdev->file_size = 0;

    }

    have_barriers = blkdev->bs->drv && blkdev->bs->drv->bdrv_flush ? 1 : 0;



    xen_be_printf(xendev, 1, "type \"%s\", fileproto \"%s\", filename \"%s\","

		  " size %" PRId64 " (%" PRId64 " MB)\n",

		  blkdev->type, blkdev->fileproto, blkdev->filename,

		  blkdev->file_size, blkdev->file_size >> 20);



    /* fill info */

    xenstore_write_be_int(&blkdev->xendev, "feature-barrier", have_barriers);

    xenstore_write_be_int(&blkdev->xendev, "info",            info);

    xenstore_write_be_int(&blkdev->xendev, "sector-size",     blkdev->file_blk);

    xenstore_write_be_int(&blkdev->xendev, "sectors",

			  blkdev->file_size / blkdev->file_blk);

    return 0;

}
