static int load_multiboot(void *fw_cfg,

                          FILE *f,

                          const char *kernel_filename,

                          const char *initrd_filename,

                          const char *kernel_cmdline,

                          uint8_t *header)

{

    int i, t, is_multiboot = 0;

    uint32_t flags = 0;

    uint32_t mh_entry_addr;

    uint32_t mh_load_addr;

    uint32_t mb_kernel_size;

    uint32_t mmap_addr = MULTIBOOT_STRUCT_ADDR;

    uint32_t mb_bootinfo = MULTIBOOT_STRUCT_ADDR + 0x500;

    uint32_t mb_cmdline = mb_bootinfo + 0x200;

    uint32_t mb_mod_end;



    /* Ok, let's see if it is a multiboot image.

       The header is 12x32bit long, so the latest entry may be 8192 - 48. */

    for (i = 0; i < (8192 - 48); i += 4) {

        if (ldl_p(header+i) == 0x1BADB002) {

            uint32_t checksum = ldl_p(header+i+8);

            flags = ldl_p(header+i+4);

            checksum += flags;

            checksum += (uint32_t)0x1BADB002;

            if (!checksum) {

                is_multiboot = 1;

                break;

            }

        }

    }



    if (!is_multiboot)

        return 0; /* no multiboot */



#ifdef DEBUG_MULTIBOOT

    fprintf(stderr, "qemu: I believe we found a multiboot image!\n");

#endif



    if (flags & 0x00000004) { /* MULTIBOOT_HEADER_HAS_VBE */

        fprintf(stderr, "qemu: multiboot knows VBE. we don't.\n");

    }

    if (!(flags & 0x00010000)) { /* MULTIBOOT_HEADER_HAS_ADDR */

        uint64_t elf_entry;

        int kernel_size;

        fclose(f);

        kernel_size = load_elf(kernel_filename, 0, &elf_entry, NULL, NULL,

                               0, ELF_MACHINE, 0);

        if (kernel_size < 0) {

            fprintf(stderr, "Error while loading elf kernel\n");

            exit(1);

        }

        mh_load_addr = mh_entry_addr = elf_entry;

        mb_kernel_size = kernel_size;



#ifdef DEBUG_MULTIBOOT

        fprintf(stderr, "qemu: loading multiboot-elf kernel (%#x bytes) with entry %#zx\n",

                mb_kernel_size, (size_t)mh_entry_addr);

#endif

    } else {

        /* Valid if mh_flags sets MULTIBOOT_HEADER_HAS_ADDR. */

        uint32_t mh_header_addr = ldl_p(header+i+12);

        mh_load_addr = ldl_p(header+i+16);

#ifdef DEBUG_MULTIBOOT

        uint32_t mh_load_end_addr = ldl_p(header+i+20);

        uint32_t mh_bss_end_addr = ldl_p(header+i+24);

#endif

        uint32_t mb_kernel_text_offset = i - (mh_header_addr - mh_load_addr);



        mh_entry_addr = ldl_p(header+i+28);

        mb_kernel_size = get_file_size(f) - mb_kernel_text_offset;



        /* Valid if mh_flags sets MULTIBOOT_HEADER_HAS_VBE.

        uint32_t mh_mode_type = ldl_p(header+i+32);

        uint32_t mh_width = ldl_p(header+i+36);

        uint32_t mh_height = ldl_p(header+i+40);

        uint32_t mh_depth = ldl_p(header+i+44); */



#ifdef DEBUG_MULTIBOOT

        fprintf(stderr, "multiboot: mh_header_addr = %#x\n", mh_header_addr);

        fprintf(stderr, "multiboot: mh_load_addr = %#x\n", mh_load_addr);

        fprintf(stderr, "multiboot: mh_load_end_addr = %#x\n", mh_load_end_addr);

        fprintf(stderr, "multiboot: mh_bss_end_addr = %#x\n", mh_bss_end_addr);

#endif



        fseek(f, mb_kernel_text_offset, SEEK_SET);



#ifdef DEBUG_MULTIBOOT

        fprintf(stderr, "qemu: loading multiboot kernel (%#x bytes) at %#x\n",

                mb_kernel_size, mh_load_addr);

#endif



        if (!fread_targphys_ok(mh_load_addr, mb_kernel_size, f)) {

            fprintf(stderr, "qemu: read error on multiboot kernel '%s' (%#x)\n",

                    kernel_filename, mb_kernel_size);

            exit(1);

        }

        fclose(f);

    }



    /* blob size is only the kernel for now */

    mb_mod_end = mh_load_addr + mb_kernel_size;



    /* load modules */

    stl_phys(mb_bootinfo + 20, 0x0); /* mods_count */

    if (initrd_filename) {

        uint32_t mb_mod_info = mb_bootinfo + 0x100;

        uint32_t mb_mod_cmdline = mb_bootinfo + 0x300;

        uint32_t mb_mod_start = mh_load_addr;

        uint32_t mb_mod_length = mb_kernel_size;

        char *next_initrd;

        char *next_space;

        int mb_mod_count = 0;



        do {

            next_initrd = strchr(initrd_filename, ',');

            if (next_initrd)

                *next_initrd = '\0';

            /* if a space comes after the module filename, treat everything

               after that as parameters */

            cpu_physical_memory_write(mb_mod_cmdline, (uint8_t*)initrd_filename,

                                      strlen(initrd_filename) + 1);

            stl_phys(mb_mod_info + 8, mb_mod_cmdline); /* string */

            mb_mod_cmdline += strlen(initrd_filename) + 1;

            if ((next_space = strchr(initrd_filename, ' ')))

                *next_space = '\0';

#ifdef DEBUG_MULTIBOOT

            printf("multiboot loading module: %s\n", initrd_filename);

#endif

            f = fopen(initrd_filename, "rb");

            if (f) {

                mb_mod_start = (mb_mod_start + mb_mod_length + (TARGET_PAGE_SIZE - 1))

                             & (TARGET_PAGE_MASK);

                mb_mod_length = get_file_size(f);

                mb_mod_end = mb_mod_start + mb_mod_length;



                if (!fread_targphys_ok(mb_mod_start, mb_mod_length, f)) {

                    fprintf(stderr, "qemu: read error on multiboot module '%s' (%#x)\n",

                            initrd_filename, mb_mod_length);

                    exit(1);

                }



                mb_mod_count++;

                stl_phys(mb_mod_info + 0, mb_mod_start);

                stl_phys(mb_mod_info + 4, mb_mod_start + mb_mod_length);

#ifdef DEBUG_MULTIBOOT

                printf("mod_start: %#x\nmod_end:   %#x\n", mb_mod_start,

                       mb_mod_start + mb_mod_length);

#endif

                stl_phys(mb_mod_info + 12, 0x0); /* reserved */

            }

            initrd_filename = next_initrd+1;

            mb_mod_info += 16;

        } while (next_initrd);

        stl_phys(mb_bootinfo + 20, mb_mod_count); /* mods_count */

        stl_phys(mb_bootinfo + 24, mb_bootinfo + 0x100); /* mods_addr */

    }



    /* Make sure we're getting kernel + modules back after reset */

    option_rom_setup_reset(mh_load_addr, mb_mod_end - mh_load_addr);



    /* Commandline support */

    stl_phys(mb_bootinfo + 16, mb_cmdline);

    t = strlen(kernel_filename);

    cpu_physical_memory_write(mb_cmdline, (uint8_t*)kernel_filename, t);

    mb_cmdline += t;

    stb_phys(mb_cmdline++, ' ');

    t = strlen(kernel_cmdline) + 1;

    cpu_physical_memory_write(mb_cmdline, (uint8_t*)kernel_cmdline, t);



    /* the kernel is where we want it to be now */



#define MULTIBOOT_FLAGS_MEMORY (1 << 0)

#define MULTIBOOT_FLAGS_BOOT_DEVICE (1 << 1)

#define MULTIBOOT_FLAGS_CMDLINE (1 << 2)

#define MULTIBOOT_FLAGS_MODULES (1 << 3)

#define MULTIBOOT_FLAGS_MMAP (1 << 6)

    stl_phys(mb_bootinfo, MULTIBOOT_FLAGS_MEMORY

                        | MULTIBOOT_FLAGS_BOOT_DEVICE

                        | MULTIBOOT_FLAGS_CMDLINE

                        | MULTIBOOT_FLAGS_MODULES

                        | MULTIBOOT_FLAGS_MMAP);

    stl_phys(mb_bootinfo + 4, 640); /* mem_lower */

    stl_phys(mb_bootinfo + 8, ram_size / 1024); /* mem_upper */

    stl_phys(mb_bootinfo + 12, 0x8001ffff); /* XXX: use the -boot switch? */

    stl_phys(mb_bootinfo + 48, mmap_addr); /* mmap_addr */



#ifdef DEBUG_MULTIBOOT

    fprintf(stderr, "multiboot: mh_entry_addr = %#x\n", mh_entry_addr);

#endif



    /* Pass variables to option rom */

    fw_cfg_add_i32(fw_cfg, FW_CFG_KERNEL_ADDR, mh_entry_addr);

    fw_cfg_add_i32(fw_cfg, FW_CFG_INITRD_ADDR, mb_bootinfo);

    fw_cfg_add_i32(fw_cfg, FW_CFG_INITRD_SIZE, mmap_addr);



    /* Make sure we're getting the config space back after reset */

    option_rom_setup_reset(mb_bootinfo, 0x500);



    option_rom[nb_option_roms] = "multiboot.bin";

    nb_option_roms++;



    return 1; /* yes, we are multiboot */

}
