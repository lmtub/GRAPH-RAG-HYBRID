static void io_mem_init(void)

{

    cpu_register_io_memory(IO_MEM_ROM >> IO_MEM_SHIFT, error_mem_read, unassigned_mem_write, NULL);

    cpu_register_io_memory(IO_MEM_UNASSIGNED >> IO_MEM_SHIFT, unassigned_mem_read, unassigned_mem_write, NULL);

    cpu_register_io_memory(IO_MEM_NOTDIRTY >> IO_MEM_SHIFT, error_mem_read, notdirty_mem_write, NULL);

    io_mem_nb = 5;







    /* alloc dirty bits array */

    phys_ram_dirty = qemu_vmalloc(phys_ram_size >> TARGET_PAGE_BITS);

    memset(phys_ram_dirty, 0xff, phys_ram_size >> TARGET_PAGE_BITS);

}