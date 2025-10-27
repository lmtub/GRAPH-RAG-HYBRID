ram_addr_t qemu_ram_alloc(ram_addr_t size, MemoryRegion *mr, Error **errp)

{

    return qemu_ram_alloc_from_ptr(size, NULL, mr, errp);

}
