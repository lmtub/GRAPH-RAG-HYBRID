PcGuestInfo *pc_guest_info_init(ram_addr_t below_4g_mem_size,

                                ram_addr_t above_4g_mem_size)

{

    PcGuestInfoState *guest_info_state = g_malloc0(sizeof *guest_info_state);

    PcGuestInfo *guest_info = &guest_info_state->info;



    guest_info->pci_info.w32.end = IO_APIC_DEFAULT_ADDRESS;

    if (sizeof(hwaddr) == 4) {

        guest_info->pci_info.w64.begin = 0;

        guest_info->pci_info.w64.end = 0;

    } else {

        /*

         * BIOS does not set MTRR entries for the 64 bit window, so no need to

         * align address to power of two.  Align address at 1G, this makes sure

         * it can be exactly covered with a PAT entry even when using huge

         * pages.

         */

        guest_info->pci_info.w64.begin =

            ROUND_UP((0x1ULL << 32) + above_4g_mem_size, 0x1ULL << 30);

        guest_info->pci_info.w64.end = guest_info->pci_info.w64.begin +

            (0x1ULL << 62);

        assert(guest_info->pci_info.w64.begin <= guest_info->pci_info.w64.end);

    }



    guest_info_state->machine_done.notify = pc_guest_info_machine_done;

    qemu_add_machine_init_done_notifier(&guest_info_state->machine_done);

    return guest_info;

}
