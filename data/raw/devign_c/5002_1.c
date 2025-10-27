void *qemu_memalign(size_t alignment, size_t size)

{

    if (!size) {

        abort();

    }

    return oom_check(VirtualAlloc(NULL, size, MEM_COMMIT, PAGE_READWRITE));

}
