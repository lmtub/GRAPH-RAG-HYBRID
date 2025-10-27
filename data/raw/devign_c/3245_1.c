static int pick_geometry(FDrive *drv)

{

    BlockBackend *blk = drv->blk;

    const FDFormat *parse;

    uint64_t nb_sectors, size;

    int i;

    int match, size_match, type_match;

    bool magic = drv->drive == FLOPPY_DRIVE_TYPE_AUTO;



    /* We can only pick a geometry if we have a diskette. */

    if (!drv->blk || !blk_is_inserted(drv->blk) ||

        drv->drive == FLOPPY_DRIVE_TYPE_NONE)

    {

        return -1;

    }



    /* We need to determine the likely geometry of the inserted medium.

     * In order of preference, we look for:

     * (1) The same drive type and number of sectors,

     * (2) The same diskette size and number of sectors,

     * (3) The same drive type.

     *

     * In all cases, matches that occur higher in the drive table will take

     * precedence over matches that occur later in the table.

     */

    blk_get_geometry(blk, &nb_sectors);

    match = size_match = type_match = -1;

    for (i = 0; ; i++) {

        parse = &fd_formats[i];

        if (parse->drive == FLOPPY_DRIVE_TYPE_NONE) {

            break;

        }

        size = (parse->max_head + 1) * parse->max_track * parse->last_sect;

        if (nb_sectors == size) {

            if (magic || parse->drive == drv->drive) {

                /* (1) perfect match -- nb_sectors and drive type */

                goto out;

            } else if (drive_size(parse->drive) == drive_size(drv->drive)) {

                /* (2) size match -- nb_sectors and physical medium size */

                match = (match == -1) ? i : match;

            } else {

                /* This is suspicious -- Did the user misconfigure? */

                size_match = (size_match == -1) ? i : size_match;

            }

        } else if (type_match == -1) {

            if ((parse->drive == drv->drive) ||

                (magic && (parse->drive == get_fallback_drive_type(drv)))) {

                /* (3) type match -- nb_sectors mismatch, but matches the type

                 *     specified explicitly by the user, or matches the fallback

                 *     default type when using the drive autodetect mechanism */

                type_match = i;

            }

        }

    }



    /* No exact match found */

    if (match == -1) {

        if (size_match != -1) {

            parse = &fd_formats[size_match];

            FLOPPY_DPRINTF("User requested floppy drive type '%s', "

                           "but inserted medium appears to be a "

                           "%d sector '%s' type\n",

                           FloppyDriveType_lookup[drv->drive],

                           nb_sectors,

                           FloppyDriveType_lookup[parse->drive]);

        }

        match = type_match;

    }



    /* No match of any kind found -- fd_format is misconfigured, abort. */

    if (match == -1) {

        error_setg(&error_abort, "No candidate geometries present in table "

                   " for floppy drive type '%s'",

                   FloppyDriveType_lookup[drv->drive]);

    }



    parse = &(fd_formats[match]);



 out:

    if (parse->max_head == 0) {

        drv->flags &= ~FDISK_DBL_SIDES;

    } else {

        drv->flags |= FDISK_DBL_SIDES;

    }

    drv->max_track = parse->max_track;

    drv->last_sect = parse->last_sect;

    drv->disk = parse->drive;

    drv->media_rate = parse->rate;

    return 0;

}
