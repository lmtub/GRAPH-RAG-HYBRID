void *qemu_try_blockalign(BlockDriverState *bs, size_t size)

{

    size_t align = bdrv_opt_mem_align(bs);



    /* Ensure that NULL is never returned on success */

    assert(align > 0);

    if (size == 0) {

        size = align;

    }



    return qemu_try_memalign(align, size);

}
