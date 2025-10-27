static Qcow2BitmapList *bitmap_list_load(BlockDriverState *bs, uint64_t offset,

                                         uint64_t size, Error **errp)

{

    int ret;

    BDRVQcow2State *s = bs->opaque;

    uint8_t *dir, *dir_end;

    Qcow2BitmapDirEntry *e;

    uint32_t nb_dir_entries = 0;

    Qcow2BitmapList *bm_list = NULL;



    if (size == 0) {

        error_setg(errp, "Requested bitmap directory size is zero");

        return NULL;

    }



    if (size > QCOW2_MAX_BITMAP_DIRECTORY_SIZE) {

        error_setg(errp, "Requested bitmap directory size is too big");

        return NULL;

    }



    dir = g_try_malloc(size);

    if (dir == NULL) {

        error_setg(errp, "Failed to allocate space for bitmap directory");

        return NULL;

    }

    dir_end = dir + size;



    ret = bdrv_pread(bs->file, offset, dir, size);

    if (ret < 0) {

        error_setg_errno(errp, -ret, "Failed to read bitmap directory");

        goto fail;

    }



    bm_list = bitmap_list_new();

    for (e = (Qcow2BitmapDirEntry *)dir;

         e < (Qcow2BitmapDirEntry *)dir_end;

         e = next_dir_entry(e))

    {

        Qcow2Bitmap *bm;



        if ((uint8_t *)(e + 1) > dir_end) {

            goto broken_dir;

        }



        if (++nb_dir_entries > s->nb_bitmaps) {

            error_setg(errp, "More bitmaps found than specified in header"

                       " extension");

            goto fail;

        }

        bitmap_dir_entry_to_cpu(e);



        if ((uint8_t *)next_dir_entry(e) > dir_end) {

            goto broken_dir;

        }



        if (e->extra_data_size != 0) {

            error_setg(errp, "Bitmap extra data is not supported");

            goto fail;

        }



        ret = check_dir_entry(bs, e);

        if (ret < 0) {

            error_setg(errp, "Bitmap '%.*s' doesn't satisfy the constraints",

                       e->name_size, dir_entry_name_field(e));

            goto fail;

        }



        bm = g_new(Qcow2Bitmap, 1);

        bm->table.offset = e->bitmap_table_offset;

        bm->table.size = e->bitmap_table_size;

        bm->flags = e->flags;

        bm->granularity_bits = e->granularity_bits;

        bm->name = dir_entry_copy_name(e);

        QSIMPLEQ_INSERT_TAIL(bm_list, bm, entry);

    }



    if (nb_dir_entries != s->nb_bitmaps) {

        error_setg(errp, "Less bitmaps found than specified in header"

                         " extension");

        goto fail;

    }



    if ((uint8_t *)e != dir_end) {

        goto broken_dir;

    }



    g_free(dir);

    return bm_list;



broken_dir:

    ret = -EINVAL;

    error_setg(errp, "Broken bitmap directory");



fail:

    g_free(dir);

    bitmap_list_free(bm_list);



    return NULL;

}
