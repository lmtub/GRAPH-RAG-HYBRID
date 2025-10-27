void *qemu_blockalign(BlockDriverState *bs, size_t size)

{

    return qemu_memalign(bdrv_opt_mem_align(bs), size);

}
