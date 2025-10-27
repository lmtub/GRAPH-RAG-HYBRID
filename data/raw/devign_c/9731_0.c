static int vvfat_open(BlockDriverState *bs, QDict *options, int flags,

                      Error **errp)

{

    BDRVVVFATState *s = bs->opaque;

    int cyls, heads, secs;

    bool floppy;

    const char *dirname, *label;

    QemuOpts *opts;

    Error *local_err = NULL;

    int ret;



#ifdef DEBUG

    vvv = s;

#endif



    opts = qemu_opts_create(&runtime_opts, NULL, 0, &error_abort);

    qemu_opts_absorb_qdict(opts, options, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        ret = -EINVAL;

        goto fail;

    }



    dirname = qemu_opt_get(opts, "dir");

    if (!dirname) {

        error_setg(errp, "vvfat block driver requires a 'dir' option");

        ret = -EINVAL;

        goto fail;

    }



    s->fat_type = qemu_opt_get_number(opts, "fat-type", 0);

    floppy = qemu_opt_get_bool(opts, "floppy", false);



    memset(s->volume_label, ' ', sizeof(s->volume_label));

    label = qemu_opt_get(opts, "label");

    if (label) {

        size_t label_length = strlen(label);

        if (label_length > 11) {

            error_setg(errp, "vvfat label cannot be longer than 11 bytes");

            ret = -EINVAL;

            goto fail;

        }

        memcpy(s->volume_label, label, label_length);

    } else {

        memcpy(s->volume_label, "QEMU VVFAT", 10);

    }



    if (floppy) {

        /* 1.44MB or 2.88MB floppy.  2.88MB can be FAT12 (default) or FAT16. */

        if (!s->fat_type) {

            s->fat_type = 12;

            secs = 36;

            s->sectors_per_cluster = 2;

        } else {

            secs = s->fat_type == 12 ? 18 : 36;

            s->sectors_per_cluster = 1;

        }

        cyls = 80;

        heads = 2;

    } else {

        /* 32MB or 504MB disk*/

        if (!s->fat_type) {

            s->fat_type = 16;

        }

        s->offset_to_bootsector = 0x3f;

        cyls = s->fat_type == 12 ? 64 : 1024;

        heads = 16;

        secs = 63;

    }



    switch (s->fat_type) {

    case 32:

            fprintf(stderr, "Big fat greek warning: FAT32 has not been tested. "

                "You are welcome to do so!\n");

        break;

    case 16:

    case 12:

        break;

    default:

        error_setg(errp, "Valid FAT types are only 12, 16 and 32");

        ret = -EINVAL;

        goto fail;

    }





    s->bs = bs;



    /* LATER TODO: if FAT32, adjust */

    s->sectors_per_cluster=0x10;



    s->current_cluster=0xffffffff;



    s->qcow = NULL;

    s->qcow_filename = NULL;

    s->fat2 = NULL;

    s->downcase_short_names = 1;



    fprintf(stderr, "vvfat %s chs %d,%d,%d\n",

            dirname, cyls, heads, secs);



    s->sector_count = cyls * heads * secs - s->offset_to_bootsector;



    if (qemu_opt_get_bool(opts, "rw", false)) {

        if (!bdrv_is_read_only(bs)) {

            ret = enable_write_target(bs, errp);

            if (ret < 0) {

                goto fail;

            }

        } else {

            ret = -EPERM;

            error_setg(errp,

                       "Unable to set VVFAT to 'rw' when drive is read-only");

            goto fail;

        }

    } else  {

        /* read only is the default for safety */

        ret = bdrv_set_read_only(bs, true, &local_err);

        if (ret < 0) {

            error_propagate(errp, local_err);

            goto fail;

        }

    }



    bs->total_sectors = cyls * heads * secs;



    if (init_directories(s, dirname, heads, secs, errp)) {

        ret = -EIO;

        goto fail;

    }



    s->sector_count = s->offset_to_root_dir

                    + s->sectors_per_cluster * s->cluster_count;



    /* Disable migration when vvfat is used rw */

    if (s->qcow) {

        error_setg(&s->migration_blocker,

                   "The vvfat (rw) format used by node '%s' "

                   "does not support live migration",

                   bdrv_get_device_or_node_name(bs));

        ret = migrate_add_blocker(s->migration_blocker, &local_err);

        if (local_err) {

            error_propagate(errp, local_err);

            error_free(s->migration_blocker);

            goto fail;

        }

    }



    if (s->offset_to_bootsector > 0) {

        init_mbr(s, cyls, heads, secs);

    }



    qemu_co_mutex_init(&s->lock);



    ret = 0;

fail:

    qemu_opts_del(opts);

    return ret;

}
