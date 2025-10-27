void helper_slbie(CPUPPCState *env, target_ulong addr)

{

    PowerPCCPU *cpu = ppc_env_get_cpu(env);

    ppc_slb_t *slb;



    slb = slb_lookup(cpu, addr);

    if (!slb) {

        return;

    }



    if (slb->esid & SLB_ESID_V) {

        slb->esid &= ~SLB_ESID_V;



        /* XXX: given the fact that segment size is 256 MB or 1TB,

         *      and we still don't have a tlb_flush_mask(env, n, mask)

         *      in QEMU, we just invalidate all TLBs

         */

        tlb_flush(CPU(cpu), 1);

    }

}
