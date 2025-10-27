static inline int ppcemb_tlb_check(CPUState *env, ppcemb_tlb_t *tlb,

                                   target_phys_addr_t *raddrp,

                                   target_ulong address, uint32_t pid, int ext,

                                   int i)

{

    target_ulong mask;



    /* Check valid flag */

    if (!(tlb->prot & PAGE_VALID)) {

        qemu_log("%s: TLB %d not valid\n", __func__, i);

        return -1;

    }

    mask = ~(tlb->size - 1);

    LOG_SWTLB("%s: TLB %d address " TARGET_FMT_lx " PID %u <=> " TARGET_FMT_lx

              " " TARGET_FMT_lx " %u\n", __func__, i, address, pid, tlb->EPN,

              mask, (uint32_t)tlb->PID);

    /* Check PID */

    if (tlb->PID != 0 && tlb->PID != pid)

        return -1;

    /* Check effective address */

    if ((address & mask) != tlb->EPN)

        return -1;

    *raddrp = (tlb->RPN & mask) | (address & ~mask);

#if (TARGET_PHYS_ADDR_BITS >= 36)

    if (ext) {

        /* Extend the physical address to 36 bits */

        *raddrp |= (target_phys_addr_t)(tlb->RPN & 0xF) << 32;

    }

#endif



    return 0;

}
