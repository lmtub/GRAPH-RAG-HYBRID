static void raven_realize(PCIDevice *d, Error **errp)

{

    RavenPCIState *s = RAVEN_PCI_DEVICE(d);

    char *filename;

    int bios_size = -1;



    d->config[0x0C] = 0x08; // cache_line_size

    d->config[0x0D] = 0x10; // latency_timer

    d->config[0x34] = 0x00; // capabilities_pointer



    memory_region_init_ram(&s->bios, OBJECT(s), "bios", BIOS_SIZE,

                           &error_abort);

    memory_region_set_readonly(&s->bios, true);

    memory_region_add_subregion(get_system_memory(), (uint32_t)(-BIOS_SIZE),

                                &s->bios);

    vmstate_register_ram_global(&s->bios);

    if (s->bios_name) {

        filename = qemu_find_file(QEMU_FILE_TYPE_BIOS, s->bios_name);

        if (filename) {

            if (s->elf_machine != EM_NONE) {

                bios_size = load_elf(filename, NULL, NULL, NULL,

                                     NULL, NULL, 1, s->elf_machine, 0);

            }

            if (bios_size < 0) {

                bios_size = get_image_size(filename);

                if (bios_size > 0 && bios_size <= BIOS_SIZE) {

                    hwaddr bios_addr;

                    bios_size = (bios_size + 0xfff) & ~0xfff;

                    bios_addr = (uint32_t)(-BIOS_SIZE);

                    bios_size = load_image_targphys(filename, bios_addr,

                                                    bios_size);

                }

            }

        }

        if (bios_size < 0 || bios_size > BIOS_SIZE) {

            hw_error("qemu: could not load bios image '%s'\n", s->bios_name);

        }

        g_free(filename);

    }

}
