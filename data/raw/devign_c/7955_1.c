void *qemu_malloc(size_t size)

{

    if (!size && !allow_zero_malloc()) {

        abort();

    }

    return oom_check(malloc(size ? size : 1));

}
