static void *file_ram_alloc(RAMBlock *block,

                            ram_addr_t memory,

                            const char *path,

                            Error **errp)

{

    char *filename;

    char *sanitized_name;

    char *c;

    void *area;

    int fd;

    uint64_t hpagesize;

    Error *local_err = NULL;



    hpagesize = gethugepagesize(path, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        goto error;

    }

    block->mr->align = hpagesize;



    if (memory < hpagesize) {

        error_setg(errp, "memory size 0x" RAM_ADDR_FMT " must be equal to "

                   "or larger than huge page size 0x%" PRIx64,

                   memory, hpagesize);

        goto error;

    }



    if (kvm_enabled() && !kvm_has_sync_mmu()) {

        error_setg(errp,

                   "host lacks kvm mmu notifiers, -mem-path unsupported");

        goto error;

    }



    /* Make name safe to use with mkstemp by replacing '/' with '_'. */

    sanitized_name = g_strdup(memory_region_name(block->mr));

    for (c = sanitized_name; *c != '\0'; c++) {

        if (*c == '/')

            *c = '_';

    }



    filename = g_strdup_printf("%s/qemu_back_mem.%s.XXXXXX", path,

                               sanitized_name);

    g_free(sanitized_name);



    fd = mkstemp(filename);

    if (fd < 0) {

        error_setg_errno(errp, errno,

                         "unable to create backing store for hugepages");

        g_free(filename);

        goto error;

    }

    unlink(filename);

    g_free(filename);



    memory = ROUND_UP(memory, hpagesize);



    /*

     * ftruncate is not supported by hugetlbfs in older

     * hosts, so don't bother bailing out on errors.

     * If anything goes wrong with it under other filesystems,

     * mmap will fail.

     */

    if (ftruncate(fd, memory)) {

        perror("ftruncate");

    }



    area = qemu_ram_mmap(fd, memory, hpagesize, block->flags & RAM_SHARED);

    if (area == MAP_FAILED) {

        error_setg_errno(errp, errno,

                         "unable to map backing store for hugepages");

        close(fd);

        goto error;

    }



    if (mem_prealloc) {

        os_mem_prealloc(fd, area, memory);

    }



    block->fd = fd;

    return area;



error:

    return NULL;

}
