void helper_store_msr(CPUPPCState *env, target_ulong val)

{

    uint32_t excp = hreg_store_msr(env, val, 0);



    if (excp != 0) {

        CPUState *cs = CPU(ppc_env_get_cpu(env));

        cs->interrupt_request |= CPU_INTERRUPT_EXITTB;

        raise_exception(env, excp);

    }

}
