static void blkdebug_refresh_filename(BlockDriverState *bs)

{

    QDict *opts;

    const QDictEntry *e;

    bool force_json = false;



    for (e = qdict_first(bs->options); e; e = qdict_next(bs->options, e)) {

        if (strcmp(qdict_entry_key(e), "config") &&

            strcmp(qdict_entry_key(e), "x-image") &&

            strcmp(qdict_entry_key(e), "image") &&

            strncmp(qdict_entry_key(e), "image.", strlen("image.")))

        {

            force_json = true;

            break;

        }

    }



    if (force_json && !bs->file->bs->full_open_options) {

        /* The config file cannot be recreated, so creating a plain filename

         * is impossible */

        return;

    }



    if (!force_json && bs->file->bs->exact_filename[0]) {

        snprintf(bs->exact_filename, sizeof(bs->exact_filename),

                 "blkdebug:%s:%s",

                 qdict_get_try_str(bs->options, "config") ?: "",

                 bs->file->bs->exact_filename);

    }



    opts = qdict_new();

    qdict_put_obj(opts, "driver", QOBJECT(qstring_from_str("blkdebug")));



    QINCREF(bs->file->bs->full_open_options);

    qdict_put_obj(opts, "image", QOBJECT(bs->file->bs->full_open_options));



    for (e = qdict_first(bs->options); e; e = qdict_next(bs->options, e)) {

        if (strcmp(qdict_entry_key(e), "x-image") &&

            strcmp(qdict_entry_key(e), "image") &&

            strncmp(qdict_entry_key(e), "image.", strlen("image.")))

        {

            qobject_incref(qdict_entry_value(e));

            qdict_put_obj(opts, qdict_entry_key(e), qdict_entry_value(e));

        }

    }



    bs->full_open_options = opts;

}
