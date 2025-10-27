static void trigger_access_exception(CPUS390XState *env, uint32_t type,

                                     uint32_t ilen, uint64_t tec)

{

    S390CPU *cpu = s390_env_get_cpu(env);



    if (kvm_enabled()) {

        kvm_s390_access_exception(cpu, type, tec);

    } else {

        CPUState *cs = CPU(cpu);

        stq_phys(cs->as, env->psa + offsetof(LowCore, trans_exc_code), tec);

        trigger_pgm_exception(env, type, ilen);

    }

}
