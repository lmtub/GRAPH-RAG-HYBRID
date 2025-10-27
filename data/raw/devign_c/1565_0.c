void r4k_helper_tlbr(CPUMIPSState *env)

{

    r4k_tlb_t *tlb;

    uint8_t ASID;

    int idx;



    ASID = env->CP0_EntryHi & 0xFF;

    idx = (env->CP0_Index & ~0x80000000) % env->tlb->nb_tlb;

    tlb = &env->tlb->mmu.r4k.tlb[idx];



    /* If this will change the current ASID, flush qemu's TLB.  */

    if (ASID != tlb->ASID)

        cpu_mips_tlb_flush (env, 1);



    r4k_mips_tlb_flush_extra(env, env->tlb->nb_tlb);



    env->CP0_EntryHi = tlb->VPN | tlb->ASID;

    env->CP0_PageMask = tlb->PageMask;

    env->CP0_EntryLo0 = tlb->G | (tlb->V0 << 1) | (tlb->D0 << 2) |

                        ((target_ulong)tlb->RI0 << CP0EnLo_RI) |

                        ((target_ulong)tlb->XI0 << CP0EnLo_XI) |

                        (tlb->C0 << 3) | (tlb->PFN[0] >> 6);

    env->CP0_EntryLo1 = tlb->G | (tlb->V1 << 1) | (tlb->D1 << 2) |

                        ((target_ulong)tlb->RI1 << CP0EnLo_RI) |

                        ((target_ulong)tlb->XI1 << CP0EnLo_XI) |

                        (tlb->C1 << 3) | (tlb->PFN[1] >> 6);

}
