static int create_shared_memory_BAR(IVShmemState *s, int fd, uint8_t attr,

                                    Error **errp)

{

    void * ptr;



    ptr = mmap(0, s->ivshmem_size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);

    if (ptr == MAP_FAILED) {

        error_setg_errno(errp, errno, "Failed to mmap shared memory");

        return -1;

    }



    s->shm_fd = fd;



    memory_region_init_ram_ptr(&s->ivshmem, OBJECT(s), "ivshmem.bar2",

                               s->ivshmem_size, ptr);

    vmstate_register_ram(&s->ivshmem, DEVICE(s));

    memory_region_add_subregion(&s->bar, 0, &s->ivshmem);



    /* region for shared memory */

    pci_register_bar(PCI_DEVICE(s), 2, attr, &s->bar);



    return 0;

}
