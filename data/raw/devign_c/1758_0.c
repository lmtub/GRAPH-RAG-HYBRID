static BlockDriverState *bdrv_open_inherit(const char *filename,

                                           const char *reference,

                                           QDict *options, int flags,

                                           BlockDriverState *parent,

                                           const BdrvChildRole *child_role,

                                           Error **errp)

{

    int ret;

    BdrvChild *file = NULL;

    BlockDriverState *bs;

    BlockDriver *drv = NULL;

    const char *drvname;

    const char *backing;

    Error *local_err = NULL;

    QDict *snapshot_options = NULL;

    int snapshot_flags = 0;



    assert(!child_role || !flags);

    assert(!child_role == !parent);



    if (reference) {

        bool options_non_empty = options ? qdict_size(options) : false;

        QDECREF(options);



        if (filename || options_non_empty) {

            error_setg(errp, "Cannot reference an existing block device with "

                       "additional options or a new filename");

            return NULL;

        }



        bs = bdrv_lookup_bs(reference, reference, errp);

        if (!bs) {

            return NULL;

        }



        bdrv_ref(bs);

        return bs;

    }



    bs = bdrv_new();



    /* NULL means an empty set of options */

    if (options == NULL) {

        options = qdict_new();

    }



    /* json: syntax counts as explicit options, as if in the QDict */

    parse_json_protocol(options, &filename, &local_err);

    if (local_err) {

        goto fail;

    }



    bs->explicit_options = qdict_clone_shallow(options);



    if (child_role) {

        bs->inherits_from = parent;

        child_role->inherit_options(&flags, options,

                                    parent->open_flags, parent->options);

    }



    ret = bdrv_fill_options(&options, filename, &flags, &local_err);

    if (local_err) {

        goto fail;

    }



    /* Set the BDRV_O_RDWR and BDRV_O_ALLOW_RDWR flags.

     * FIXME: we're parsing the QDict to avoid having to create a

     * QemuOpts just for this, but neither option is optimal. */

    if (g_strcmp0(qdict_get_try_str(options, BDRV_OPT_READ_ONLY), "on") &&

        !qdict_get_try_bool(options, BDRV_OPT_READ_ONLY, false)) {

        flags |= (BDRV_O_RDWR | BDRV_O_ALLOW_RDWR);

    } else {

        flags &= ~BDRV_O_RDWR;

    }



    if (flags & BDRV_O_SNAPSHOT) {

        snapshot_options = qdict_new();

        bdrv_temp_snapshot_options(&snapshot_flags, snapshot_options,

                                   flags, options);

        /* Let bdrv_backing_options() override "read-only" */

        qdict_del(options, BDRV_OPT_READ_ONLY);

        bdrv_backing_options(&flags, options, flags, options);

    }



    bs->open_flags = flags;

    bs->options = options;

    options = qdict_clone_shallow(options);



    /* Find the right image format driver */

    drvname = qdict_get_try_str(options, "driver");

    if (drvname) {

        drv = bdrv_find_format(drvname);

        if (!drv) {

            error_setg(errp, "Unknown driver: '%s'", drvname);

            goto fail;

        }

    }



    assert(drvname || !(flags & BDRV_O_PROTOCOL));



    backing = qdict_get_try_str(options, "backing");

    if (backing && *backing == '\0') {

        flags |= BDRV_O_NO_BACKING;

        qdict_del(options, "backing");

    }



    /* Open image file without format layer */

    if ((flags & BDRV_O_PROTOCOL) == 0) {

        file = bdrv_open_child(filename, options, "file", bs,

                               &child_file, true, &local_err);

        if (local_err) {

            goto fail;

        }

    }



    /* Image format probing */

    bs->probed = !drv;

    if (!drv && file) {

        ret = find_image_format(file, filename, &drv, &local_err);

        if (ret < 0) {

            goto fail;

        }

        /*

         * This option update would logically belong in bdrv_fill_options(),

         * but we first need to open bs->file for the probing to work, while

         * opening bs->file already requires the (mostly) final set of options

         * so that cache mode etc. can be inherited.

         *

         * Adding the driver later is somewhat ugly, but it's not an option

         * that would ever be inherited, so it's correct. We just need to make

         * sure to update both bs->options (which has the full effective

         * options for bs) and options (which has file.* already removed).

         */

        qdict_put(bs->options, "driver", qstring_from_str(drv->format_name));

        qdict_put(options, "driver", qstring_from_str(drv->format_name));

    } else if (!drv) {

        error_setg(errp, "Must specify either driver or file");

        goto fail;

    }



    /* BDRV_O_PROTOCOL must be set iff a protocol BDS is about to be created */

    assert(!!(flags & BDRV_O_PROTOCOL) == !!drv->bdrv_file_open);

    /* file must be NULL if a protocol BDS is about to be created

     * (the inverse results in an error message from bdrv_open_common()) */

    assert(!(flags & BDRV_O_PROTOCOL) || !file);



    /* Open the image */

    ret = bdrv_open_common(bs, file, options, &local_err);

    if (ret < 0) {

        goto fail;

    }



    if (file && (bs->file != file)) {

        bdrv_unref_child(bs, file);

        file = NULL;

    }



    /* If there is a backing file, use it */

    if ((flags & BDRV_O_NO_BACKING) == 0) {

        ret = bdrv_open_backing_file(bs, options, "backing", &local_err);

        if (ret < 0) {

            goto close_and_fail;

        }

    }



    bdrv_refresh_filename(bs);



    /* Check if any unknown options were used */

    if (options && (qdict_size(options) != 0)) {

        const QDictEntry *entry = qdict_first(options);

        if (flags & BDRV_O_PROTOCOL) {

            error_setg(errp, "Block protocol '%s' doesn't support the option "

                       "'%s'", drv->format_name, entry->key);

        } else {

            error_setg(errp,

                       "Block format '%s' does not support the option '%s'",

                       drv->format_name, entry->key);

        }



        goto close_and_fail;

    }



    if (!bdrv_key_required(bs)) {

        bdrv_parent_cb_change_media(bs, true);

    } else if (!runstate_check(RUN_STATE_PRELAUNCH)

               && !runstate_check(RUN_STATE_INMIGRATE)

               && !runstate_check(RUN_STATE_PAUSED)) { /* HACK */

        error_setg(errp,

                   "Guest must be stopped for opening of encrypted image");

        goto close_and_fail;

    }



    QDECREF(options);



    /* For snapshot=on, create a temporary qcow2 overlay. bs points to the

     * temporary snapshot afterwards. */

    if (snapshot_flags) {

        BlockDriverState *snapshot_bs;

        snapshot_bs = bdrv_append_temp_snapshot(bs, snapshot_flags,

                                                snapshot_options, &local_err);

        snapshot_options = NULL;

        if (local_err) {

            goto close_and_fail;

        }

        /* We are not going to return bs but the overlay on top of it

         * (snapshot_bs); thus, we have to drop the strong reference to bs

         * (which we obtained by calling bdrv_new()). bs will not be deleted,

         * though, because the overlay still has a reference to it. */

        bdrv_unref(bs);

        bs = snapshot_bs;

    }



    return bs;



fail:

    if (file != NULL) {

        bdrv_unref_child(bs, file);

    }

    QDECREF(snapshot_options);

    QDECREF(bs->explicit_options);

    QDECREF(bs->options);

    QDECREF(options);

    bs->options = NULL;

    bdrv_unref(bs);

    error_propagate(errp, local_err);

    return NULL;



close_and_fail:

    bdrv_unref(bs);

    QDECREF(snapshot_options);

    QDECREF(options);

    error_propagate(errp, local_err);

    return NULL;

}
