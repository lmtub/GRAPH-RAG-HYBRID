void kqemu_set_phys_mem(uint64_t start_addr, ram_addr_t size, 

                        ram_addr_t phys_offset)

{

    struct kqemu_phys_mem kphys_mem1, *kphys_mem = &kphys_mem1;

    uint64_t end;

    int ret, io_index;



    end = (start_addr + size + TARGET_PAGE_SIZE - 1) & TARGET_PAGE_MASK;

    start_addr &= TARGET_PAGE_MASK;

    kphys_mem->phys_addr = start_addr;

    kphys_mem->size = end - start_addr;

    kphys_mem->ram_addr = phys_offset & TARGET_PAGE_MASK;

    io_index = phys_offset & ~TARGET_PAGE_MASK;

    switch(io_index) {

    case IO_MEM_RAM:

        kphys_mem->io_index = KQEMU_IO_MEM_RAM;

        break;

    case IO_MEM_ROM:

        kphys_mem->io_index = KQEMU_IO_MEM_ROM;

        break;

    default:

        if (qpi_io_memory == io_index) {

            kphys_mem->io_index = KQEMU_IO_MEM_COMM;

        } else {

            kphys_mem->io_index = KQEMU_IO_MEM_UNASSIGNED;

        }

        break;

    }

#ifdef _WIN32

    {

        DWORD temp;

        ret = DeviceIoControl(kqemu_fd, KQEMU_SET_PHYS_MEM, 

                              kphys_mem, sizeof(*kphys_mem),

                              NULL, 0, &temp, NULL) == TRUE ? 0 : -1;

    }

#else

    ret = ioctl(kqemu_fd, KQEMU_SET_PHYS_MEM, kphys_mem);

#endif

    if (ret < 0) {

        fprintf(stderr, "kqemu: KQEMU_SET_PHYS_PAGE error=%d: start_addr=0x%016" PRIx64 " size=0x%08lx phys_offset=0x%08lx\n",

                ret, start_addr, 

                (unsigned long)size, (unsigned long)phys_offset);

    }

}
