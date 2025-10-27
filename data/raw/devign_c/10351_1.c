void tlb_set_page_with_attrs(CPUState *cpu, target_ulong vaddr,

                             hwaddr paddr, MemTxAttrs attrs, int prot,

                             int mmu_idx, target_ulong size)

{

    CPUArchState *env = cpu->env_ptr;

    MemoryRegionSection *section;

    unsigned int index;

    target_ulong address;

    target_ulong code_address;

    uintptr_t addend;

    CPUTLBEntry *te;

    hwaddr iotlb, xlat, sz;

    unsigned vidx = env->vtlb_index++ % CPU_VTLB_SIZE;

    int asidx = cpu_asidx_from_attrs(cpu, attrs);



    assert_cpu_is_self(cpu);

    assert(size >= TARGET_PAGE_SIZE);

    if (size != TARGET_PAGE_SIZE) {

        tlb_add_large_page(env, vaddr, size);

    }



    sz = size;

    section = address_space_translate_for_iotlb(cpu, asidx, paddr, &xlat, &sz);

    assert(sz >= TARGET_PAGE_SIZE);



    tlb_debug("vaddr=" TARGET_FMT_lx " paddr=0x" TARGET_FMT_plx

              " prot=%x idx=%d\n",

              vaddr, paddr, prot, mmu_idx);



    address = vaddr;

    if (!memory_region_is_ram(section->mr) && !memory_region_is_romd(section->mr)) {

        /* IO memory case */

        address |= TLB_MMIO;

        addend = 0;

    } else {

        /* TLB_MMIO for rom/romd handled below */

        addend = (uintptr_t)memory_region_get_ram_ptr(section->mr) + xlat;

    }



    code_address = address;

    iotlb = memory_region_section_get_iotlb(cpu, section, vaddr, paddr, xlat,

                                            prot, &address);



    index = (vaddr >> TARGET_PAGE_BITS) & (CPU_TLB_SIZE - 1);

    te = &env->tlb_table[mmu_idx][index];



    /* do not discard the translation in te, evict it into a victim tlb */

    env->tlb_v_table[mmu_idx][vidx] = *te;

    env->iotlb_v[mmu_idx][vidx] = env->iotlb[mmu_idx][index];



    /* refill the tlb */

    env->iotlb[mmu_idx][index].addr = iotlb - vaddr;

    env->iotlb[mmu_idx][index].attrs = attrs;

    te->addend = addend - vaddr;

    if (prot & PAGE_READ) {

        te->addr_read = address;

    } else {

        te->addr_read = -1;

    }



    if (prot & PAGE_EXEC) {

        te->addr_code = code_address;

    } else {

        te->addr_code = -1;

    }

    if (prot & PAGE_WRITE) {

        if ((memory_region_is_ram(section->mr) && section->readonly)

            || memory_region_is_romd(section->mr)) {

            /* Write access calls the I/O callback.  */

            te->addr_write = address | TLB_MMIO;

        } else if (memory_region_is_ram(section->mr)

                   && cpu_physical_memory_is_clean(

                        memory_region_get_ram_addr(section->mr) + xlat)) {

            te->addr_write = address | TLB_NOTDIRTY;

        } else {

            te->addr_write = address;

        }

    } else {

        te->addr_write = -1;

    }

}
