static void tlb_info_pae32(Monitor *mon, CPUState *env)

{

    int l1, l2, l3;

    uint64_t pdpe, pde, pte;

    uint64_t pdp_addr, pd_addr, pt_addr;



    pdp_addr = env->cr[3] & ~0x1f;

    for (l1 = 0; l1 < 4; l1++) {

        cpu_physical_memory_read(pdp_addr + l1 * 8, &pdpe, 8);

        pdpe = le64_to_cpu(pdpe);

        if (pdpe & PG_PRESENT_MASK) {

            pd_addr = pdpe & 0x3fffffffff000ULL;

            for (l2 = 0; l2 < 512; l2++) {

                cpu_physical_memory_read(pd_addr + l2 * 8, &pde, 8);

                pde = le64_to_cpu(pde);

                if (pde & PG_PRESENT_MASK) {

                    if (pde & PG_PSE_MASK) {

                        /* 2M pages with PAE, CR4.PSE is ignored */

                        print_pte(mon, (l1 << 30 ) + (l2 << 21), pde,

                                  ~((target_phys_addr_t)(1 << 20) - 1));

                    } else {

                        pt_addr = pde & 0x3fffffffff000ULL;

                        for (l3 = 0; l3 < 512; l3++) {

                            cpu_physical_memory_read(pt_addr + l3 * 8, &pte, 8);

                            pte = le64_to_cpu(pte);

                            if (pte & PG_PRESENT_MASK) {

                                print_pte(mon, (l1 << 30 ) + (l2 << 21)

                                          + (l3 << 12),

                                          pte & ~PG_PSE_MASK,

                                          ~(target_phys_addr_t)0xfff);

                            }

                        }

                    }

                }

            }

        }

    }

}
