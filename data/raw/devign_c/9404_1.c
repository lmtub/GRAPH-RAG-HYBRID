static int mmubooke206_get_physical_address(CPUState *env, mmu_ctx_t *ctx,

                                            target_ulong address, int rw,

                                            int access_type)

{

    ppcmas_tlb_t *tlb;

    target_phys_addr_t raddr;

    int i, j, ret;



    ret = -1;

    raddr = (target_phys_addr_t)-1ULL;



    for (i = 0; i < BOOKE206_MAX_TLBN; i++) {

        int ways = booke206_tlb_ways(env, i);



        for (j = 0; j < ways; j++) {

            tlb = booke206_get_tlbm(env, i, address, j);




            ret = mmubooke206_check_tlb(env, tlb, &raddr, &ctx->prot, address,

                                        rw, access_type);

            if (ret != -1) {

                goto found_tlb;






found_tlb:



    if (ret >= 0) {

        ctx->raddr = raddr;

        LOG_SWTLB("%s: access granted " TARGET_FMT_lx " => " TARGET_FMT_plx

                  " %d %d\n", __func__, address, ctx->raddr, ctx->prot,

                  ret);

    } else {

        LOG_SWTLB("%s: access refused " TARGET_FMT_lx " => " TARGET_FMT_plx

                  " %d %d\n", __func__, address, raddr, ctx->prot, ret);




    return ret;
