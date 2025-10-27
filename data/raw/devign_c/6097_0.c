static int bdrv_fill_options(QDict **options, const char **pfilename, int flags,

                             BlockDriver *drv, Error **errp)

{

    const char *filename = *pfilename;

    const char *drvname;

    bool protocol = flags & BDRV_O_PROTOCOL;

    bool parse_filename = false;

    Error *local_err = NULL;



    /* Parse json: pseudo-protocol */

    if (filename && g_str_has_prefix(filename, "json:")) {

        QDict *json_options = parse_json_filename(filename, &local_err);

        if (local_err) {

            error_propagate(errp, local_err);

            return -EINVAL;

        }



        /* Options given in the filename have lower priority than options

         * specified directly */

        qdict_join(*options, json_options, false);

        QDECREF(json_options);

        *pfilename = filename = NULL;

    }



    /* Fetch the file name from the options QDict if necessary */

    if (protocol && filename) {

        if (!qdict_haskey(*options, "filename")) {

            qdict_put(*options, "filename", qstring_from_str(filename));

            parse_filename = true;

        } else {

            error_setg(errp, "Can't specify 'file' and 'filename' options at "

                             "the same time");

            return -EINVAL;

        }

    }



    /* Find the right block driver */

    filename = qdict_get_try_str(*options, "filename");

    drvname = qdict_get_try_str(*options, "driver");



    if (drv) {

        if (drvname) {

            error_setg(errp, "Driver specified twice");

            return -EINVAL;

        }

        drvname = drv->format_name;

        qdict_put(*options, "driver", qstring_from_str(drvname));

    } else {

        if (!drvname && protocol) {

            if (filename) {

                drv = bdrv_find_protocol(filename, parse_filename, errp);

                if (!drv) {

                    return -EINVAL;

                }



                drvname = drv->format_name;

                qdict_put(*options, "driver", qstring_from_str(drvname));

            } else {

                error_setg(errp, "Must specify either driver or file");

                return -EINVAL;

            }

        } else if (drvname) {

            drv = bdrv_find_format(drvname);

            if (!drv) {

                error_setg(errp, "Unknown driver '%s'", drvname);

                return -ENOENT;

            }

        }

    }



    assert(drv || !protocol);



    /* Driver-specific filename parsing */

    if (drv && drv->bdrv_parse_filename && parse_filename) {

        drv->bdrv_parse_filename(filename, *options, &local_err);

        if (local_err) {

            error_propagate(errp, local_err);

            return -EINVAL;

        }



        if (!drv->bdrv_needs_filename) {

            qdict_del(*options, "filename");

        }

    }



    return 0;

}
