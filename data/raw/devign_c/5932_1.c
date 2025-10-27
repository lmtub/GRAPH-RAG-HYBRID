static int parallels_open(BlockDriverState *bs, QDict *options, int flags,

                          Error **errp)

{

    BDRVParallelsState *s = bs->opaque;

    ParallelsHeader ph;

    int ret, size, i;

    QemuOpts *opts = NULL;

    Error *local_err = NULL;

    char *buf;



    bs->file = bdrv_open_child(NULL, options, "file", bs, &child_file,

                               false, errp);

    if (!bs->file) {

        return -EINVAL;

    }



    ret = bdrv_pread(bs->file, 0, &ph, sizeof(ph));

    if (ret < 0) {

        goto fail;

    }



    bs->total_sectors = le64_to_cpu(ph.nb_sectors);



    if (le32_to_cpu(ph.version) != HEADER_VERSION) {

        goto fail_format;

    }

    if (!memcmp(ph.magic, HEADER_MAGIC, 16)) {

        s->off_multiplier = 1;

        bs->total_sectors = 0xffffffff & bs->total_sectors;

    } else if (!memcmp(ph.magic, HEADER_MAGIC2, 16)) {

        s->off_multiplier = le32_to_cpu(ph.tracks);

    } else {

        goto fail_format;

    }



    s->tracks = le32_to_cpu(ph.tracks);

    if (s->tracks == 0) {

        error_setg(errp, "Invalid image: Zero sectors per track");

        ret = -EINVAL;

        goto fail;

    }

    if (s->tracks > INT32_MAX/513) {

        error_setg(errp, "Invalid image: Too big cluster");

        ret = -EFBIG;

        goto fail;

    }



    s->bat_size = le32_to_cpu(ph.bat_entries);

    if (s->bat_size > INT_MAX / sizeof(uint32_t)) {

        error_setg(errp, "Catalog too large");

        ret = -EFBIG;

        goto fail;

    }



    size = bat_entry_off(s->bat_size);

    s->header_size = ROUND_UP(size, bdrv_opt_mem_align(bs->file->bs));

    s->header = qemu_try_blockalign(bs->file->bs, s->header_size);

    if (s->header == NULL) {

        ret = -ENOMEM;

        goto fail;

    }

    s->data_end = le32_to_cpu(ph.data_off);

    if (s->data_end == 0) {

        s->data_end = ROUND_UP(bat_entry_off(s->bat_size), BDRV_SECTOR_SIZE);

    }

    if (s->data_end < s->header_size) {

        /* there is not enough unused space to fit to block align between BAT

           and actual data. We can't avoid read-modify-write... */

        s->header_size = size;

    }



    ret = bdrv_pread(bs->file, 0, s->header, s->header_size);

    if (ret < 0) {

        goto fail;

    }

    s->bat_bitmap = (uint32_t *)(s->header + 1);



    for (i = 0; i < s->bat_size; i++) {

        int64_t off = bat2sect(s, i);

        if (off >= s->data_end) {

            s->data_end = off + s->tracks;

        }

    }



    if (le32_to_cpu(ph.inuse) == HEADER_INUSE_MAGIC) {

        /* Image was not closed correctly. The check is mandatory */

        s->header_unclean = true;

        if ((flags & BDRV_O_RDWR) && !(flags & BDRV_O_CHECK)) {

            error_setg(errp, "parallels: Image was not closed correctly; "

                       "cannot be opened read/write");

            ret = -EACCES;

            goto fail;

        }

    }



    opts = qemu_opts_create(&parallels_runtime_opts, NULL, 0, &local_err);

    if (local_err != NULL) {

        goto fail_options;

    }



    qemu_opts_absorb_qdict(opts, options, &local_err);

    if (local_err != NULL) {

        goto fail_options;

    }



    s->prealloc_size =

        qemu_opt_get_size_del(opts, PARALLELS_OPT_PREALLOC_SIZE, 0);

    s->prealloc_size = MAX(s->tracks, s->prealloc_size >> BDRV_SECTOR_BITS);

    buf = qemu_opt_get_del(opts, PARALLELS_OPT_PREALLOC_MODE);

    s->prealloc_mode = qapi_enum_parse(prealloc_mode_lookup, buf,

            PRL_PREALLOC_MODE__MAX, PRL_PREALLOC_MODE_FALLOCATE, &local_err);

    g_free(buf);

    if (local_err != NULL) {

        goto fail_options;

    }



    if (!(flags & BDRV_O_RESIZE) || !bdrv_has_zero_init(bs->file->bs) ||

            bdrv_truncate(bs->file, bdrv_getlength(bs->file->bs),

                          PREALLOC_MODE_OFF, NULL) != 0) {

        s->prealloc_mode = PRL_PREALLOC_MODE_FALLOCATE;

    }



    if (flags & BDRV_O_RDWR) {

        s->header->inuse = cpu_to_le32(HEADER_INUSE_MAGIC);

        ret = parallels_update_header(bs);

        if (ret < 0) {

            goto fail;

        }

    }



    s->bat_dirty_block = 4 * getpagesize();

    s->bat_dirty_bmap =

        bitmap_new(DIV_ROUND_UP(s->header_size, s->bat_dirty_block));



    qemu_co_mutex_init(&s->lock);

    return 0;



fail_format:

    error_setg(errp, "Image not in Parallels format");

    ret = -EINVAL;

fail:

    qemu_vfree(s->header);

    return ret;



fail_options:

    error_propagate(errp, local_err);

    ret = -EINVAL;

    goto fail;

}
