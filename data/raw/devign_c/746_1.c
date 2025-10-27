static int vdi_check(BlockDriverState *bs)

{

    /* TODO: additional checks possible. */

    BDRVVdiState *s = (BDRVVdiState *)bs->opaque;

    int n_errors = 0;

    uint32_t blocks_allocated = 0;

    uint32_t block;

    uint32_t *bmap;

    logout("\n");



    bmap = qemu_malloc(s->header.blocks_in_image * sizeof(uint32_t));

    memset(bmap, 0xff, s->header.blocks_in_image * sizeof(uint32_t));



    /* Check block map and value of blocks_allocated. */

    for (block = 0; block < s->header.blocks_in_image; block++) {

        uint32_t bmap_entry = le32_to_cpu(s->bmap[block]);

        if (bmap_entry != VDI_UNALLOCATED) {

            if (bmap_entry < s->header.blocks_in_image) {

                blocks_allocated++;

                if (bmap[bmap_entry] == VDI_UNALLOCATED) {

                    bmap[bmap_entry] = bmap_entry;

                } else {

                    fprintf(stderr, "ERROR: block index %" PRIu32

                            " also used by %" PRIu32 "\n", bmap[bmap_entry], bmap_entry);

                }

            } else {

                fprintf(stderr, "ERROR: block index %" PRIu32

                        " too large, is %" PRIu32 "\n", block, bmap_entry);

                n_errors++;

            }

        }

    }

    if (blocks_allocated != s->header.blocks_allocated) {

        fprintf(stderr, "ERROR: allocated blocks mismatch, is %" PRIu32

               ", should be %" PRIu32 "\n",

               blocks_allocated, s->header.blocks_allocated);

        n_errors++;

    }



    qemu_free(bmap);



    return n_errors;

}
