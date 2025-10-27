BdrvChild *bdrv_open_child(const char *filename,

                           QDict *options, const char *bdref_key,

                           BlockDriverState* parent,

                           const BdrvChildRole *child_role,

                           bool allow_none, Error **errp)

{

    BdrvChild *c = NULL;

    BlockDriverState *bs;

    QDict *image_options;

    int ret;

    char *bdref_key_dot;

    const char *reference;



    assert(child_role != NULL);



    bdref_key_dot = g_strdup_printf("%s.", bdref_key);

    qdict_extract_subqdict(options, &image_options, bdref_key_dot);

    g_free(bdref_key_dot);



    reference = qdict_get_try_str(options, bdref_key);

    if (!filename && !reference && !qdict_size(image_options)) {

        if (!allow_none) {

            error_setg(errp, "A block device must be specified for \"%s\"",

                       bdref_key);

        }

        QDECREF(image_options);

        goto done;

    }



    bs = NULL;

    ret = bdrv_open_inherit(&bs, filename, reference, image_options, 0,

                            parent, child_role, errp);

    if (ret < 0) {

        goto done;

    }



    c = bdrv_attach_child(parent, bs, child_role);



done:

    qdict_del(options, bdref_key);

    return c;

}
