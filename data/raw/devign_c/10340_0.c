static void parse_drive(DeviceState *dev, const char *str, void **ptr,

                        const char *propname, Error **errp)

{

    BlockBackend *blk;



    blk = blk_by_name(str);

    if (!blk) {

        error_setg(errp, "Property '%s.%s' can't find value '%s'",

                   object_get_typename(OBJECT(dev)), propname, str);

        return;

    }

    if (blk_attach_dev(blk, dev) < 0) {

        DriveInfo *dinfo = blk_legacy_dinfo(blk);



        if (dinfo->type != IF_NONE) {

            error_setg(errp, "Drive '%s' is already in use because "

                       "it has been automatically connected to another "

                       "device (did you need 'if=none' in the drive options?)",

                       str);

        } else {

            error_setg(errp, "Drive '%s' is already in use by another device",

                       str);

        }

        return;

    }

    *ptr = blk;

}
