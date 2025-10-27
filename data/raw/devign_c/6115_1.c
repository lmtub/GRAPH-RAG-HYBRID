int bdrv_file_open(BlockDriverState **pbs, const char *filename,

                   const char *reference, QDict *options, int flags,

                   Error **errp)

{

    BlockDriverState *bs = NULL;

    BlockDriver *drv;

    const char *drvname;

    bool allow_protocol_prefix = false;

    Error *local_err = NULL;

    int ret;



    /* NULL means an empty set of options */

    if (options == NULL) {

        options = qdict_new();

    }



    if (reference) {

        if (filename || qdict_size(options)) {

            error_setg(errp, "Cannot reference an existing block device with "

                       "additional options or a new filename");

            return -EINVAL;

        }

        QDECREF(options);



        bs = bdrv_find(reference);

        if (!bs) {

            error_setg(errp, "Cannot find block device '%s'", reference);

            return -ENODEV;

        }

        bdrv_ref(bs);

        *pbs = bs;

        return 0;

    }



    bs = bdrv_new("");

    bs->options = options;

    options = qdict_clone_shallow(options);



    /* Fetch the file name from the options QDict if necessary */

    if (!filename) {

        filename = qdict_get_try_str(options, "filename");

    } else if (filename && !qdict_haskey(options, "filename")) {

        qdict_put(options, "filename", qstring_from_str(filename));

        allow_protocol_prefix = true;

    } else {

        error_setg(errp, "Can't specify 'file' and 'filename' options at the "

                   "same time");

        ret = -EINVAL;

        goto fail;

    }



    /* Find the right block driver */

    drvname = qdict_get_try_str(options, "driver");

    if (drvname) {

        drv = bdrv_find_format(drvname);

        if (!drv) {

            error_setg(errp, "Unknown driver '%s'", drvname);

        }

        qdict_del(options, "driver");

    } else if (filename) {

        drv = bdrv_find_protocol(filename, allow_protocol_prefix);

        if (!drv) {

            error_setg(errp, "Unknown protocol");

        }

    } else {

        error_setg(errp, "Must specify either driver or file");

        drv = NULL;

    }



    if (!drv) {

        /* errp has been set already */

        ret = -ENOENT;

        goto fail;

    }



    /* Parse the filename and open it */

    if (drv->bdrv_parse_filename && filename) {

        drv->bdrv_parse_filename(filename, options, &local_err);

        if (error_is_set(&local_err)) {

            error_propagate(errp, local_err);

            ret = -EINVAL;

            goto fail;

        }

        qdict_del(options, "filename");

    } else if (drv->bdrv_needs_filename && !filename) {

        error_setg(errp, "The '%s' block driver requires a file name",

                   drv->format_name);

        ret = -EINVAL;

        goto fail;

    }



    if (!drv->bdrv_file_open) {

        ret = bdrv_open(bs, filename, options, flags, drv, &local_err);

        options = NULL;

    } else {

        ret = bdrv_open_common(bs, NULL, options, flags, drv, &local_err);

    }

    if (ret < 0) {

        error_propagate(errp, local_err);

        goto fail;

    }



    /* Check if any unknown options were used */

    if (options && (qdict_size(options) != 0)) {

        const QDictEntry *entry = qdict_first(options);

        error_setg(errp, "Block protocol '%s' doesn't support the option '%s'",

                   drv->format_name, entry->key);

        ret = -EINVAL;

        goto fail;

    }

    QDECREF(options);



    bs->growable = 1;

    *pbs = bs;

    return 0;



fail:

    QDECREF(options);

    if (!bs->drv) {

        QDECREF(bs->options);

    }

    bdrv_unref(bs);

    return ret;

}
