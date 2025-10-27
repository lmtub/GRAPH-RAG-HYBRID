static int sd_create(const char *filename, QEMUOptionParameter *options,

                     Error **errp)

{

    int ret = 0;

    uint32_t vid = 0;

    char *backing_file = NULL;

    BDRVSheepdogState *s;

    char tag[SD_MAX_VDI_TAG_LEN];

    uint32_t snapid;

    bool prealloc = false;

    Error *local_err = NULL;



    s = g_malloc0(sizeof(BDRVSheepdogState));



    memset(tag, 0, sizeof(tag));

    if (strstr(filename, "://")) {

        ret = sd_parse_uri(s, filename, s->name, &snapid, tag);

    } else {

        ret = parse_vdiname(s, filename, s->name, &snapid, tag);

    }

    if (ret < 0) {

        goto out;

    }



    while (options && options->name) {

        if (!strcmp(options->name, BLOCK_OPT_SIZE)) {

            s->inode.vdi_size = options->value.n;

        } else if (!strcmp(options->name, BLOCK_OPT_BACKING_FILE)) {

            backing_file = options->value.s;

        } else if (!strcmp(options->name, BLOCK_OPT_PREALLOC)) {

            if (!options->value.s || !strcmp(options->value.s, "off")) {

                prealloc = false;

            } else if (!strcmp(options->value.s, "full")) {

                prealloc = true;

            } else {

                error_report("Invalid preallocation mode: '%s'",

                             options->value.s);

                ret = -EINVAL;

                goto out;

            }

        } else if (!strcmp(options->name, BLOCK_OPT_REDUNDANCY)) {

            ret = parse_redundancy(s, options->value.s);

            if (ret < 0) {

                goto out;

            }

        }

        options++;

    }



    if (s->inode.vdi_size > SD_MAX_VDI_SIZE) {

        error_report("too big image size");

        ret = -EINVAL;

        goto out;

    }



    if (backing_file) {

        BlockDriverState *bs;

        BDRVSheepdogState *s;

        BlockDriver *drv;



        /* Currently, only Sheepdog backing image is supported. */

        drv = bdrv_find_protocol(backing_file, true);

        if (!drv || strcmp(drv->protocol_name, "sheepdog") != 0) {

            error_report("backing_file must be a sheepdog image");

            ret = -EINVAL;

            goto out;

        }



        ret = bdrv_file_open(&bs, backing_file, NULL, 0, &local_err);

        if (ret < 0) {

            qerror_report_err(local_err);

            error_free(local_err);

            goto out;

        }



        s = bs->opaque;



        if (!is_snapshot(&s->inode)) {

            error_report("cannot clone from a non snapshot vdi");

            bdrv_unref(bs);

            ret = -EINVAL;

            goto out;

        }



        bdrv_unref(bs);

    }



    ret = do_sd_create(s, &vid, 0);

    if (!prealloc || ret) {

        goto out;

    }



    ret = sd_prealloc(filename);

out:

    g_free(s);

    return ret;

}
