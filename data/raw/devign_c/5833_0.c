static void sun4m_common_init(int ram_size, int boot_device, DisplayState *ds,

                              const char *kernel_filename, const char *kernel_cmdline,

                              const char *initrd_filename, const char *cpu_model,

                              unsigned int machine)

{

    sun4m_hw_init(&hwdefs[machine], ram_size, ds, cpu_model);



    sun4m_load_kernel(hwdefs[machine].vram_size, ram_size, boot_device,

                      kernel_filename, kernel_cmdline, initrd_filename,

                      hwdefs[machine].machine_id);

}
