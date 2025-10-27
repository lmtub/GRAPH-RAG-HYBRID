void blkconf_serial(BlockConf *conf, char **serial)

{

    DriveInfo *dinfo;



    if (!*serial) {

        /* try to fall back to value set with legacy -drive serial=... */

        dinfo = drive_get_by_blockdev(conf->bs);

        if (*dinfo->serial) {

            *serial = g_strdup(dinfo->serial);

        }

    }

}
