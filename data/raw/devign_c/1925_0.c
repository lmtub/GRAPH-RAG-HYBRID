void blockdev_auto_del(BlockDriverState *bs)

{

    DriveInfo *dinfo = drive_get_by_blockdev(bs);



    if (dinfo->auto_del) {

        drive_uninit(dinfo);

    }

}
