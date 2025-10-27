void vga_init(VGAState *s)

{

    int vga_io_memory;



    register_savevm("vga", 0, 2, vga_save, vga_load, s);



    register_ioport_write(0x3c0, 16, 1, vga_ioport_write, s);



    register_ioport_write(0x3b4, 2, 1, vga_ioport_write, s);

    register_ioport_write(0x3d4, 2, 1, vga_ioport_write, s);

    register_ioport_write(0x3ba, 1, 1, vga_ioport_write, s);

    register_ioport_write(0x3da, 1, 1, vga_ioport_write, s);



    register_ioport_read(0x3c0, 16, 1, vga_ioport_read, s);



    register_ioport_read(0x3b4, 2, 1, vga_ioport_read, s);

    register_ioport_read(0x3d4, 2, 1, vga_ioport_read, s);

    register_ioport_read(0x3ba, 1, 1, vga_ioport_read, s);

    register_ioport_read(0x3da, 1, 1, vga_ioport_read, s);

    s->bank_offset = 0;



#ifdef CONFIG_BOCHS_VBE

    s->vbe_regs[VBE_DISPI_INDEX_ID] = VBE_DISPI_ID0;

    s->vbe_bank_mask = ((s->vram_size >> 16) - 1);

#if defined (TARGET_I386)

    register_ioport_read(0x1ce, 1, 2, vbe_ioport_read_index, s);

    register_ioport_read(0x1cf, 1, 2, vbe_ioport_read_data, s);



    register_ioport_write(0x1ce, 1, 2, vbe_ioport_write_index, s);

    register_ioport_write(0x1cf, 1, 2, vbe_ioport_write_data, s);



    /* old Bochs IO ports */

    register_ioport_read(0xff80, 1, 2, vbe_ioport_read_index, s);

    register_ioport_read(0xff81, 1, 2, vbe_ioport_read_data, s);



    register_ioport_write(0xff80, 1, 2, vbe_ioport_write_index, s);

    register_ioport_write(0xff81, 1, 2, vbe_ioport_write_data, s);

#else

    register_ioport_read(0x1ce, 1, 2, vbe_ioport_read_index, s);

    register_ioport_read(0x1d0, 1, 2, vbe_ioport_read_data, s);



    register_ioport_write(0x1ce, 1, 2, vbe_ioport_write_index, s);

    register_ioport_write(0x1d0, 1, 2, vbe_ioport_write_data, s);

#endif

#endif /* CONFIG_BOCHS_VBE */



    vga_io_memory = cpu_register_io_memory(0, vga_mem_read, vga_mem_write, s);

    cpu_register_physical_memory(isa_mem_base + 0x000a0000, 0x20000,

                                 vga_io_memory);


}