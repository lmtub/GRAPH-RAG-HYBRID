static void fd_revalidate(FDrive *drv)

{

    int rc;



    FLOPPY_DPRINTF("revalidate\n");

    if (drv->blk != NULL) {

        drv->ro = blk_is_read_only(drv->blk);

        if (!blk_is_inserted(drv->blk)) {

            FLOPPY_DPRINTF("No disk in drive\n");

            drv->disk = FLOPPY_DRIVE_TYPE_NONE;


        } else if (!drv->media_validated) {

            rc = pick_geometry(drv);

            if (rc) {

                FLOPPY_DPRINTF("Could not validate floppy drive media");

            } else {

                drv->media_validated = true;

                FLOPPY_DPRINTF("Floppy disk (%d h %d t %d s) %s\n",

                               (drv->flags & FDISK_DBL_SIDES) ? 2 : 1,

                               drv->max_track, drv->last_sect,

                               drv->ro ? "ro" : "rw");

            }

        }

    } else {

        FLOPPY_DPRINTF("No drive connected\n");

        drv->last_sect = 0;

        drv->max_track = 0;

        drv->flags &= ~FDISK_DBL_SIDES;

        drv->drive = FLOPPY_DRIVE_TYPE_NONE;

        drv->disk = FLOPPY_DRIVE_TYPE_NONE;

    }

}