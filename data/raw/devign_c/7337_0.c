static hwaddr ppc_hash32_htab_lookup(PowerPCCPU *cpu,

                                     target_ulong sr, target_ulong eaddr,

                                     ppc_hash_pte32_t *pte)

{

    CPUPPCState *env = &cpu->env;

    hwaddr pteg_off, pte_offset;

    hwaddr hash;

    uint32_t vsid, pgidx, ptem;



    vsid = sr & SR32_VSID;

    pgidx = (eaddr & ~SEGMENT_MASK_256M) >> TARGET_PAGE_BITS;

    hash = vsid ^ pgidx;

    ptem = (vsid << 7) | (pgidx >> 10);



    /* Page address translation */

    qemu_log_mask(CPU_LOG_MMU, "htab_base " TARGET_FMT_plx

            " htab_mask " TARGET_FMT_plx

            " hash " TARGET_FMT_plx "\n",

            env->htab_base, env->htab_mask, hash);



    /* Primary PTEG lookup */

    qemu_log_mask(CPU_LOG_MMU, "0 htab=" TARGET_FMT_plx "/" TARGET_FMT_plx

            " vsid=%" PRIx32 " ptem=%" PRIx32

            " hash=" TARGET_FMT_plx "\n",

            env->htab_base, env->htab_mask, vsid, ptem, hash);

    pteg_off = get_pteg_offset32(cpu, hash);

    pte_offset = ppc_hash32_pteg_search(cpu, pteg_off, 0, ptem, pte);

    if (pte_offset == -1) {

        /* Secondary PTEG lookup */

        qemu_log_mask(CPU_LOG_MMU, "1 htab=" TARGET_FMT_plx "/" TARGET_FMT_plx

                " vsid=%" PRIx32 " api=%" PRIx32

                " hash=" TARGET_FMT_plx "\n", env->htab_base,

                env->htab_mask, vsid, ptem, ~hash);

        pteg_off = get_pteg_offset32(cpu, ~hash);

        pte_offset = ppc_hash32_pteg_search(cpu, pteg_off, 1, ptem, pte);

    }



    return pte_offset;

}
