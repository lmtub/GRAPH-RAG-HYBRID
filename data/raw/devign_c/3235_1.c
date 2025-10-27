static int cloop_open(BlockDriverState *bs, int flags)

{

    BDRVCloopState *s = bs->opaque;

    uint32_t offsets_size, max_compressed_block_size = 1, i;



    bs->read_only = 1;



    /* read header */

    if (bdrv_pread(bs->file, 128, &s->block_size, 4) < 4) {

        goto cloop_close;

    }

    s->block_size = be32_to_cpu(s->block_size);



    if (bdrv_pread(bs->file, 128 + 4, &s->n_blocks, 4) < 4) {

        goto cloop_close;

    }

    s->n_blocks = be32_to_cpu(s->n_blocks);



    /* read offsets */

    offsets_size = s->n_blocks * sizeof(uint64_t);

    s->offsets = g_malloc(offsets_size);

    if (bdrv_pread(bs->file, 128 + 4 + 4, s->offsets, offsets_size) <

            offsets_size) {

        goto cloop_close;

    }

    for(i=0;i<s->n_blocks;i++) {

        s->offsets[i] = be64_to_cpu(s->offsets[i]);

        if (i > 0) {

            uint32_t size = s->offsets[i] - s->offsets[i - 1];

            if (size > max_compressed_block_size) {

                max_compressed_block_size = size;

            }

        }

    }



    /* initialize zlib engine */

    s->compressed_block = g_malloc(max_compressed_block_size + 1);

    s->uncompressed_block = g_malloc(s->block_size);

    if (inflateInit(&s->zstream) != Z_OK) {

        goto cloop_close;

    }

    s->current_block = s->n_blocks;



    s->sectors_per_block = s->block_size/512;

    bs->total_sectors = s->n_blocks * s->sectors_per_block;

    qemu_co_mutex_init(&s->lock);

    return 0;



cloop_close:

    return -1;

}
