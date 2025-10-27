void helper_store_sdr1(CPUPPCState *env, target_ulong val)

{

    PowerPCCPU *cpu = ppc_env_get_cpu(env);



    if (!env->external_htab) {

        if (env->spr[SPR_SDR1] != val) {

            ppc_store_sdr1(env, val);

            tlb_flush(CPU(cpu));

        }

    }

}
