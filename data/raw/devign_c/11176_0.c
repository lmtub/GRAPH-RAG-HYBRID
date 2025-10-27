static int find_real_tpr_addr(VAPICROMState *s, CPUX86State *env)

{

    target_phys_addr_t paddr;

    target_ulong addr;



    if (s->state == VAPIC_ACTIVE) {

        return 0;

    }

    /*

     * If there is no prior TPR access instruction we could analyze (which is

     * the case after resume from hibernation), we need to scan the possible

     * virtual address space for the APIC mapping.

     */

    for (addr = 0xfffff000; addr >= 0x80000000; addr -= TARGET_PAGE_SIZE) {

        paddr = cpu_get_phys_page_debug(env, addr);

        if (paddr != APIC_DEFAULT_ADDRESS) {

            continue;

        }

        s->real_tpr_addr = addr + 0x80;

        update_guest_rom_state(s);

        return 0;

    }

    return -1;

}
