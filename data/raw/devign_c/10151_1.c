static hwaddr ppc_hash64_pte_raddr(ppc_slb_t *slb, ppc_hash_pte64_t pte,

                                   target_ulong eaddr)

{

    hwaddr mask;

    int target_page_bits;

    hwaddr rpn = pte.pte1 & HPTE64_R_RPN;

    /*

     * We support 4K, 64K and 16M now

     */

    target_page_bits = ppc_hash64_page_shift(slb);

    mask = (1ULL << target_page_bits) - 1;

    return (rpn & ~mask) | (eaddr & mask);

}
