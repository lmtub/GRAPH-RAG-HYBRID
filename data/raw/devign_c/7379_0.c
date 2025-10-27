static int dmg_open(BlockDriverState *bs, QDict *options, int flags,

                    Error **errp)

{

    BDRVDMGState *s = bs->opaque;

    DmgHeaderState ds;

    uint64_t rsrc_fork_offset, rsrc_fork_length;

    uint64_t plist_xml_offset, plist_xml_length;

    int64_t offset;

    int ret;



    bs->file = bdrv_open_child(NULL, options, "file", bs, &child_file,

                               false, errp);

    if (!bs->file) {

        return -EINVAL;

    }



    block_module_load_one("dmg-bz2");

    bs->read_only = true;



    s->n_chunks = 0;

    s->offsets = s->lengths = s->sectors = s->sectorcounts = NULL;

    /* used by dmg_read_mish_block to keep track of the current I/O position */

    ds.data_fork_offset = 0;

    ds.max_compressed_size = 1;

    ds.max_sectors_per_chunk = 1;



    /* locate the UDIF trailer */

    offset = dmg_find_koly_offset(bs->file, errp);

    if (offset < 0) {

        ret = offset;

        goto fail;

    }



    /* offset of data fork (DataForkOffset) */

    ret = read_uint64(bs, offset + 0x18, &ds.data_fork_offset);

    if (ret < 0) {

        goto fail;

    } else if (ds.data_fork_offset > offset) {

        ret = -EINVAL;

        goto fail;

    }



    /* offset of resource fork (RsrcForkOffset) */

    ret = read_uint64(bs, offset + 0x28, &rsrc_fork_offset);

    if (ret < 0) {

        goto fail;

    }

    ret = read_uint64(bs, offset + 0x30, &rsrc_fork_length);

    if (ret < 0) {

        goto fail;

    }

    if (rsrc_fork_offset >= offset ||

        rsrc_fork_length > offset - rsrc_fork_offset) {

        ret = -EINVAL;

        goto fail;

    }

    /* offset of property list (XMLOffset) */

    ret = read_uint64(bs, offset + 0xd8, &plist_xml_offset);

    if (ret < 0) {

        goto fail;

    }

    ret = read_uint64(bs, offset + 0xe0, &plist_xml_length);

    if (ret < 0) {

        goto fail;

    }

    if (plist_xml_offset >= offset ||

        plist_xml_length > offset - plist_xml_offset) {

        ret = -EINVAL;

        goto fail;

    }

    ret = read_uint64(bs, offset + 0x1ec, (uint64_t *)&bs->total_sectors);

    if (ret < 0) {

        goto fail;

    }

    if (bs->total_sectors < 0) {

        ret = -EINVAL;

        goto fail;

    }

    if (rsrc_fork_length != 0) {

        ret = dmg_read_resource_fork(bs, &ds,

                                     rsrc_fork_offset, rsrc_fork_length);

        if (ret < 0) {

            goto fail;

        }

    } else if (plist_xml_length != 0) {

        ret = dmg_read_plist_xml(bs, &ds, plist_xml_offset, plist_xml_length);

        if (ret < 0) {

            goto fail;

        }

    } else {

        ret = -EINVAL;

        goto fail;

    }



    /* initialize zlib engine */

    s->compressed_chunk = qemu_try_blockalign(bs->file->bs,

                                              ds.max_compressed_size + 1);

    s->uncompressed_chunk = qemu_try_blockalign(bs->file->bs,

                                                512 * ds.max_sectors_per_chunk);

    if (s->compressed_chunk == NULL || s->uncompressed_chunk == NULL) {

        ret = -ENOMEM;

        goto fail;

    }



    if (inflateInit(&s->zstream) != Z_OK) {

        ret = -EINVAL;

        goto fail;

    }



    s->current_chunk = s->n_chunks;



    qemu_co_mutex_init(&s->lock);

    return 0;



fail:

    g_free(s->types);

    g_free(s->offsets);

    g_free(s->lengths);

    g_free(s->sectors);

    g_free(s->sectorcounts);

    qemu_vfree(s->compressed_chunk);

    qemu_vfree(s->uncompressed_chunk);

    return ret;

}
