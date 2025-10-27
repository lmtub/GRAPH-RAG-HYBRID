void *qemu_try_blockalign0(BlockDriverState *bs, size_t size)

{

    void *mem = qemu_try_blockalign(bs, size);



    if (mem) {

        memset(mem, 0, size);

    }



    return mem;

}
