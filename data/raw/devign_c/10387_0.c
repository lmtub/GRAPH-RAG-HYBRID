static void borzoi_init(int ram_size, int vga_ram_size, int boot_device,

                DisplayState *ds, const char **fd_filename, int snapshot,

                const char *kernel_filename, const char *kernel_cmdline,

                const char *initrd_filename, const char *cpu_model)

{

    spitz_common_init(ram_size, vga_ram_size, ds, kernel_filename,

                kernel_cmdline, initrd_filename, borzoi, 0x33f);

}
