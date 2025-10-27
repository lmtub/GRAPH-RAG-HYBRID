void isa_mmio_setup(MemoryRegion *mr, target_phys_addr_t size)

{

    memory_region_init_io(mr, &isa_mmio_ops, NULL, "isa-mmio", size);

}
