void ppc_slb_invalidate_all (CPUPPCState *env)

{

    target_phys_addr_t sr_base;

    uint64_t tmp64;

    int n, do_invalidate;



    do_invalidate = 0;

    sr_base = env->spr[SPR_ASR];

    for (n = 0; n < env->slb_nr; n++) {

        tmp64 = ldq_phys(sr_base);

        if (slb_is_valid(tmp64)) {

            slb_invalidate(&tmp64);

            stq_phys(sr_base, tmp64);

            /* XXX: given the fact that segment size is 256 MB or 1TB,

             *      and we still don't have a tlb_flush_mask(env, n, mask)

             *      in Qemu, we just invalidate all TLBs

             */

            do_invalidate = 1;

        }

        sr_base += 12;

    }

    if (do_invalidate)

        tlb_flush(env, 1);

}
