static int vvfat_open(BlockDriverState *bs, QDict *options, int flags)

{

    BDRVVVFATState *s = bs->opaque;

    int cyls, heads, secs;

    bool floppy;

    const char *dirname;

    QemuOpts *opts;

    Error *local_err = NULL;

    int ret;



#ifdef DEBUG

    vvv = s;

#endif



DLOG(if (stderr == NULL) {

    stderr = fopen("vvfat.log", "a");

    setbuf(stderr, NULL);

})



    opts = qemu_opts_create_nofail(&runtime_opts);

    qemu_opts_absorb_qdict(opts, options, &local_err);

    if (error_is_set(&local_err)) {

        qerror_report_err(local_err);

        error_free(local_err);

        ret = -EINVAL;

        goto fail;

    }



    dirname = qemu_opt_get(opts, "dir");

    if (!dirname) {

        qerror_report(ERROR_CLASS_GENERIC_ERROR, "vvfat block driver requires "

                      "a 'dir' option");

        ret = -EINVAL;

        goto fail;

    }



    s->fat_type = qemu_opt_get_number(opts, "fat-type", 0);

    floppy = qemu_opt_get_bool(opts, "floppy", false);



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

        s->first_sectors_number = 1;

        cyls = 80;

        heads = 2;

    } else {

        /* 32MB or 504MB disk*/

        if (!s->fat_type) {

            s->fat_type = 16;

        }

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

        qerror_report(ERROR_CLASS_GENERIC_ERROR, "Valid FAT types are only "

                      "12, 16 and 32");

        ret = -EINVAL;

        goto fail;

    }





    s->bs = bs;



    /* LATER TODO: if FAT32, adjust */

    s->sectors_per_cluster=0x10;



    s->current_cluster=0xffffffff;



    s->first_sectors_number=0x40;

    /* read only is the default for safety */

    bs->read_only = 1;

    s->qcow = s->write_target = NULL;

    s->qcow_filename = NULL;

    s->fat2 = NULL;

    s->downcase_short_names = 1;



    fprintf(stderr, "vvfat %s chs %d,%d,%d\n",

            dirname, cyls, heads, secs);



    s->sector_count = cyls * heads * secs - (s->first_sectors_number - 1);



    if (qemu_opt_get_bool(opts, "rw", false)) {

        if (enable_write_target(s)) {

            ret = -EIO;

            goto fail;

        }

        bs->read_only = 0;

    }



    bs->total_sectors = cyls * heads * secs;



    if (init_directories(s, dirname, heads, secs)) {

        ret = -EIO;

        goto fail;

    }



    s->sector_count = s->faked_sectors + s->sectors_per_cluster*s->cluster_count;



    if (s->first_sectors_number == 0x40) {

        init_mbr(s, cyls, heads, secs);

    }



    //    assert(is_consistent(s));

    qemu_co_mutex_init(&s->lock);



    /* Disable migration when vvfat is used rw */

    if (s->qcow) {

        error_set(&s->migration_blocker,

                  QERR_BLOCK_FORMAT_FEATURE_NOT_SUPPORTED,

                  "vvfat (rw)", bs->device_name, "live migration");

        migrate_add_blocker(s->migration_blocker);

    }



    ret = 0;

fail:

    qemu_opts_del(opts);

    return ret;

}
