void bdrv_refresh_filename(BlockDriverState *bs)

{

    BlockDriver *drv = bs->drv;

    QDict *opts;



    if (!drv) {

        return;

    }



    /* This BDS's file name will most probably depend on its file's name, so

     * refresh that first */

    if (bs->file) {

        bdrv_refresh_filename(bs->file->bs);

    }



    if (drv->bdrv_refresh_filename) {

        /* Obsolete information is of no use here, so drop the old file name

         * information before refreshing it */

        bs->exact_filename[0] = '\0';

        if (bs->full_open_options) {

            QDECREF(bs->full_open_options);

            bs->full_open_options = NULL;

        }



        drv->bdrv_refresh_filename(bs);

    } else if (bs->file) {

        /* Try to reconstruct valid information from the underlying file */

        bool has_open_options;



        bs->exact_filename[0] = '\0';

        if (bs->full_open_options) {

            QDECREF(bs->full_open_options);

            bs->full_open_options = NULL;

        }



        opts = qdict_new();

        has_open_options = append_open_options(opts, bs);



        /* If no specific options have been given for this BDS, the filename of

         * the underlying file should suffice for this one as well */

        if (bs->file->bs->exact_filename[0] && !has_open_options) {

            strcpy(bs->exact_filename, bs->file->bs->exact_filename);

        }

        /* Reconstructing the full options QDict is simple for most format block

         * drivers, as long as the full options are known for the underlying

         * file BDS. The full options QDict of that file BDS should somehow

         * contain a representation of the filename, therefore the following

         * suffices without querying the (exact_)filename of this BDS. */

        if (bs->file->bs->full_open_options) {

            qdict_put_obj(opts, "driver",

                          QOBJECT(qstring_from_str(drv->format_name)));

            QINCREF(bs->file->bs->full_open_options);

            qdict_put_obj(opts, "file",

                          QOBJECT(bs->file->bs->full_open_options));



            bs->full_open_options = opts;

        } else {

            QDECREF(opts);

        }

    } else if (!bs->full_open_options && qdict_size(bs->options)) {

        /* There is no underlying file BDS (at least referenced by BDS.file),

         * so the full options QDict should be equal to the options given

         * specifically for this block device when it was opened (plus the

         * driver specification).

         * Because those options don't change, there is no need to update

         * full_open_options when it's already set. */



        opts = qdict_new();

        append_open_options(opts, bs);

        qdict_put_obj(opts, "driver",

                      QOBJECT(qstring_from_str(drv->format_name)));



        if (bs->exact_filename[0]) {

            /* This may not work for all block protocol drivers (some may

             * require this filename to be parsed), but we have to find some

             * default solution here, so just include it. If some block driver

             * does not support pure options without any filename at all or

             * needs some special format of the options QDict, it needs to

             * implement the driver-specific bdrv_refresh_filename() function.

             */

            qdict_put_obj(opts, "filename",

                          QOBJECT(qstring_from_str(bs->exact_filename)));

        }



        bs->full_open_options = opts;

    }



    if (bs->exact_filename[0]) {

        pstrcpy(bs->filename, sizeof(bs->filename), bs->exact_filename);

    } else if (bs->full_open_options) {

        QString *json = qobject_to_json(QOBJECT(bs->full_open_options));

        snprintf(bs->filename, sizeof(bs->filename), "json:%s",

                 qstring_get_str(json));

        QDECREF(json);

    }

}
