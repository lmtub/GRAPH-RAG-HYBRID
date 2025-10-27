static int cloop_open(BlockDriverState *bs, QDict *options, int flags,

                      Error **errp)

{

    BDRVCloopState *s = bs->opaque;

    uint32_t offsets_size, max_compressed_block_size = 1, i;

    int ret;



    bs->file = bdrv_open_child(NULL, options, "file", bs, &child_file,

                               false, errp);

    if (!bs->file) {

        return -EINVAL;

    }



    bdrv_set_read_only(bs, true);



    /* read header */

    ret = bdrv_pread(bs->file, 128, &s->block_size, 4);

    if (ret < 0) {

        return ret;

    }

    s->block_size = be32_to_cpu(s->block_size);

    if (s->block_size % 512) {

        error_setg(errp, "block_size %" PRIu32 " must be a multiple of 512",

                   s->block_size);

        return -EINVAL;

    }

    if (s->block_size == 0) {

        error_setg(errp, "block_size cannot be zero");

        return -EINVAL;

    }



    /* cloop's create_compressed_fs.c warns about block sizes beyond 256 KB but

     * we can accept more.  Prevent ridiculous values like 4 GB - 1 since we

     * need a buffer this big.

     */

    if (s->block_size > MAX_BLOCK_SIZE) {

        error_setg(errp, "block_size %" PRIu32 " must be %u MB or less",

                   s->block_size,

                   MAX_BLOCK_SIZE / (1024 * 1024));

        return -EINVAL;

    }



    ret = bdrv_pread(bs->file, 128 + 4, &s->n_blocks, 4);

    if (ret < 0) {

        return ret;

    }

    s->n_blocks = be32_to_cpu(s->n_blocks);



    /* read offsets */

    if (s->n_blocks > (UINT32_MAX - 1) / sizeof(uint64_t)) {

        /* Prevent integer overflow */

        error_setg(errp, "n_blocks %" PRIu32 " must be %zu or less",

                   s->n_blocks,

                   (UINT32_MAX - 1) / sizeof(uint64_t));

        return -EINVAL;

    }

    offsets_size = (s->n_blocks + 1) * sizeof(uint64_t);

    if (offsets_size > 512 * 1024 * 1024) {

        /* Prevent ridiculous offsets_size which causes memory allocation to

         * fail or overflows bdrv_pread() size.  In practice the 512 MB

         * offsets[] limit supports 16 TB images at 256 KB block size.

         */

        error_setg(errp, "image requires too many offsets, "

                   "try increasing block size");

        return -EINVAL;

    }



    s->offsets = g_try_malloc(offsets_size);

    if (s->offsets == NULL) {

        error_setg(errp, "Could not allocate offsets table");

        return -ENOMEM;

    }



    ret = bdrv_pread(bs->file, 128 + 4 + 4, s->offsets, offsets_size);

    if (ret < 0) {

        goto fail;

    }



    for (i = 0; i < s->n_blocks + 1; i++) {

        uint64_t size;



        s->offsets[i] = be64_to_cpu(s->offsets[i]);

        if (i == 0) {

            continue;

        }



        if (s->offsets[i] < s->offsets[i - 1]) {

            error_setg(errp, "offsets not monotonically increasing at "

                       "index %" PRIu32 ", image file is corrupt", i);

            ret = -EINVAL;

            goto fail;

        }



        size = s->offsets[i] - s->offsets[i - 1];



        /* Compressed blocks should be smaller than the uncompressed block size

         * but maybe compression performed poorly so the compressed block is

         * actually bigger.  Clamp down on unrealistic values to prevent

         * ridiculous s->compressed_block allocation.

         */

        if (size > 2 * MAX_BLOCK_SIZE) {

            error_setg(errp, "invalid compressed block size at index %" PRIu32

                       ", image file is corrupt", i);

            ret = -EINVAL;

            goto fail;

        }



        if (size > max_compressed_block_size) {

            max_compressed_block_size = size;

        }

    }



    /* initialize zlib engine */

    s->compressed_block = g_try_malloc(max_compressed_block_size + 1);

    if (s->compressed_block == NULL) {

        error_setg(errp, "Could not allocate compressed_block");

        ret = -ENOMEM;

        goto fail;

    }



    s->uncompressed_block = g_try_malloc(s->block_size);

    if (s->uncompressed_block == NULL) {

        error_setg(errp, "Could not allocate uncompressed_block");

        ret = -ENOMEM;

        goto fail;

    }



    if (inflateInit(&s->zstream) != Z_OK) {

        ret = -EINVAL;

        goto fail;

    }

    s->current_block = s->n_blocks;



    s->sectors_per_block = s->block_size/512;

    bs->total_sectors = s->n_blocks * s->sectors_per_block;

    qemu_co_mutex_init(&s->lock);

    return 0;



fail:

    g_free(s->offsets);

    g_free(s->compressed_block);

    g_free(s->uncompressed_block);

    return ret;

}
