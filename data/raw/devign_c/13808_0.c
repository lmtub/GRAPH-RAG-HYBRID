DriveInfo *drive_get(BlockInterfaceType type, int bus, int unit)

{

    DriveInfo *dinfo;



    /* seek interface, bus and unit */



    TAILQ_FOREACH(dinfo, &drives, next) {

        if (dinfo->type == type &&

	    dinfo->bus == bus &&

	    dinfo->unit == unit)

            return dinfo;

    }



    return NULL;

}
