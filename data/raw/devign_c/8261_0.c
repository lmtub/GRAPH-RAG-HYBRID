static void hax_process_section(MemoryRegionSection *section, uint8_t flags)

{

    MemoryRegion *mr = section->mr;

    hwaddr start_pa = section->offset_within_address_space;

    ram_addr_t size = int128_get64(section->size);

    unsigned int delta;

    uint64_t host_va;



    /* We only care about RAM and ROM regions */

    if (!memory_region_is_ram(mr)) {

        if (memory_region_is_romd(mr)) {

            /* HAXM kernel module does not support ROMD yet  */

            fprintf(stderr, "%s: Warning: Ignoring ROMD region 0x%016" PRIx64

                    "->0x%016" PRIx64 "\n", __func__, start_pa,

                    start_pa + size);

        }

        return;

    }



    /* Adjust start_pa and size so that they are page-aligned. (Cf

     * kvm_set_phys_mem() in kvm-all.c).

     */

    delta = qemu_real_host_page_size - (start_pa & ~qemu_real_host_page_mask);

    delta &= ~qemu_real_host_page_mask;

    if (delta > size) {

        return;

    }

    start_pa += delta;

    size -= delta;

    size &= qemu_real_host_page_mask;

    if (!size || (start_pa & ~qemu_real_host_page_mask)) {

        return;

    }



    host_va = (uintptr_t)memory_region_get_ram_ptr(mr)

            + section->offset_within_region + delta;

    if (memory_region_is_rom(section->mr)) {

        flags |= HAX_RAM_INFO_ROM;

    }



    /* the kernel module interface uses 32-bit sizes (but we could split...) */

    g_assert(size <= UINT32_MAX);



    hax_update_mapping(start_pa, size, host_va, flags);

}
