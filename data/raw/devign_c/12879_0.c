static int update_rom_mapping(VAPICROMState *s, CPUX86State *env, target_ulong ip)

{

    target_phys_addr_t paddr;

    uint32_t rom_state_vaddr;

    uint32_t pos, patch, offset;



    /* nothing to do if already activated */

    if (s->state == VAPIC_ACTIVE) {

        return 0;

    }



    /* bail out if ROM init code was not executed (missing ROM?) */

    if (s->state == VAPIC_INACTIVE) {

        return -1;

    }



    /* find out virtual address of the ROM */

    rom_state_vaddr = s->rom_state_paddr + (ip & 0xf0000000);

    paddr = cpu_get_phys_page_debug(env, rom_state_vaddr);

    if (paddr == -1) {

        return -1;

    }

    paddr += rom_state_vaddr & ~TARGET_PAGE_MASK;

    if (paddr != s->rom_state_paddr) {

        return -1;

    }

    read_guest_rom_state(s);

    if (memcmp(s->rom_state.signature, "kvm aPiC", 8) != 0) {

        return -1;

    }

    s->rom_state_vaddr = rom_state_vaddr;



    /* fixup addresses in ROM if needed */

    if (rom_state_vaddr == le32_to_cpu(s->rom_state.vaddr)) {

        return 0;

    }

    for (pos = le32_to_cpu(s->rom_state.fixup_start);

         pos < le32_to_cpu(s->rom_state.fixup_end);

         pos += 4) {

        cpu_physical_memory_rw(paddr + pos - s->rom_state.vaddr,

                               (void *)&offset, sizeof(offset), 0);

        offset = le32_to_cpu(offset);

        cpu_physical_memory_rw(paddr + offset, (void *)&patch,

                               sizeof(patch), 0);

        patch = le32_to_cpu(patch);

        patch += rom_state_vaddr - le32_to_cpu(s->rom_state.vaddr);

        patch = cpu_to_le32(patch);

        cpu_physical_memory_rw(paddr + offset, (void *)&patch,

                               sizeof(patch), 1);

    }

    read_guest_rom_state(s);

    s->vapic_paddr = paddr + le32_to_cpu(s->rom_state.vapic_vaddr) -

        le32_to_cpu(s->rom_state.vaddr);



    return 0;

}
