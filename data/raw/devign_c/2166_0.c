static void mem_info_pae32(Monitor *mon, CPUState *env)

{

    unsigned int l1, l2, l3;

    int prot, last_prot;

    uint64_t pdpe, pde, pte;

    uint64_t pdp_addr, pd_addr, pt_addr;

    target_phys_addr_t start, end;



    pdp_addr = env->cr[3] & ~0x1f;

    last_prot = 0;

    start = -1;

    for (l1 = 0; l1 < 4; l1++) {

        cpu_physical_memory_read(pdp_addr + l1 * 8, &pdpe, 8);

        pdpe = le64_to_cpu(pdpe);

        end = l1 << 30;

        if (pdpe & PG_PRESENT_MASK) {

            pd_addr = pdpe & 0x3fffffffff000ULL;

            for (l2 = 0; l2 < 512; l2++) {

                cpu_physical_memory_read(pd_addr + l2 * 8, &pde, 8);

                pde = le64_to_cpu(pde);

                end = (l1 << 30) + (l2 << 21);

                if (pde & PG_PRESENT_MASK) {

                    if (pde & PG_PSE_MASK) {

                        prot = pde & (PG_USER_MASK | PG_RW_MASK |

                                      PG_PRESENT_MASK);

                        mem_print(mon, &start, &last_prot, end, prot);

                    } else {

                        pt_addr = pde & 0x3fffffffff000ULL;

                        for (l3 = 0; l3 < 512; l3++) {

                            cpu_physical_memory_read(pt_addr + l3 * 8, &pte, 8);

                            pte = le64_to_cpu(pte);

                            end = (l1 << 30) + (l2 << 21) + (l3 << 12);

                            if (pte & PG_PRESENT_MASK) {

                                prot = pte & (PG_USER_MASK | PG_RW_MASK |

                                              PG_PRESENT_MASK);

                            } else {

                                prot = 0;

                            }

                            mem_print(mon, &start, &last_prot, end, prot);

                        }

                    }

                } else {

                    prot = 0;

                    mem_print(mon, &start, &last_prot, end, prot);

                }

            }

        } else {

            prot = 0;

            mem_print(mon, &start, &last_prot, end, prot);

        }

    }

    /* Flush last range */

    mem_print(mon, &start, &last_prot, (target_phys_addr_t)1 << 32, 0);

}
