void *qemu_blockalign0(BlockDriverState *bs, size_t size)

{

    return memset(qemu_blockalign(bs, size), 0, size);

}
