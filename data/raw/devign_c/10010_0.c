static int patch_hypercalls(VAPICROMState *s)

{

    hwaddr rom_paddr = s->rom_state_paddr & ROM_BLOCK_MASK;

    static const uint8_t vmcall_pattern[] = { /* vmcall */

        0xb8, 0x1, 0, 0, 0, 0xf, 0x1, 0xc1

    };

    static const uint8_t outl_pattern[] = { /* nop; outl %eax,0x7e */

        0xb8, 0x1, 0, 0, 0, 0x90, 0xe7, 0x7e

    };

    uint8_t alternates[2];

    const uint8_t *pattern;

    const uint8_t *patch;

    int patches = 0;

    off_t pos;

    uint8_t *rom;



    rom = g_malloc(s->rom_size);

    cpu_physical_memory_read(rom_paddr, rom, s->rom_size);



    for (pos = 0; pos < s->rom_size - sizeof(vmcall_pattern); pos++) {

        if (kvm_irqchip_in_kernel()) {

            pattern = outl_pattern;

            alternates[0] = outl_pattern[7];

            alternates[1] = outl_pattern[7];

            patch = &vmcall_pattern[5];

        } else {

            pattern = vmcall_pattern;

            alternates[0] = vmcall_pattern[7];

            alternates[1] = 0xd9; /* AMD's VMMCALL */

            patch = &outl_pattern[5];

        }

        if (memcmp(rom + pos, pattern, 7) == 0 &&

            (rom[pos + 7] == alternates[0] || rom[pos + 7] == alternates[1])) {

            cpu_physical_memory_write(rom_paddr + pos + 5, patch, 3);

            /*

             * Don't flush the tb here. Under ordinary conditions, the patched

             * calls are miles away from the current IP. Under malicious

             * conditions, the guest could trick us to crash.

             */

        }

    }



    g_free(rom);



    if (patches != 0 && patches != 2) {

        return -1;

    }



    return 0;

}
