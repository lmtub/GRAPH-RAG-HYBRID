static void do_ext_interrupt(CPUS390XState *env)

{

    S390CPU *cpu = s390_env_get_cpu(env);

    uint64_t mask, addr;

    LowCore *lowcore;

    ExtQueue *q;



    if (!(env->psw.mask & PSW_MASK_EXT)) {

        cpu_abort(CPU(cpu), "Ext int w/o ext mask\n");

    }



    lowcore = cpu_map_lowcore(env);



    if (env->pending_int & INTERRUPT_EXT_CLOCK_COMPARATOR) {

        lowcore->ext_int_code = cpu_to_be16(EXT_CLOCK_COMP);

        lowcore->cpu_addr = 0;

        env->pending_int &= ~INTERRUPT_EXT_CLOCK_COMPARATOR;

    } else if (env->pending_int & INTERRUPT_EXT_CPU_TIMER) {

        lowcore->ext_int_code = cpu_to_be16(EXT_CPU_TIMER);

        lowcore->cpu_addr = 0;

        env->pending_int &= ~INTERRUPT_EXT_CPU_TIMER;

    } else if (env->pending_int & INTERRUPT_EXT_SERVICE) {

        g_assert(env->ext_index >= 0);

        /*

         * FIXME: floating IRQs should be considered by all CPUs and

         *        shuld not get cleared by CPU reset.

         */

        q = &env->ext_queue[env->ext_index];

        lowcore->ext_int_code = cpu_to_be16(q->code);

        lowcore->ext_params = cpu_to_be32(q->param);

        lowcore->ext_params2 = cpu_to_be64(q->param64);

        lowcore->cpu_addr = cpu_to_be16(env->core_id | VIRTIO_SUBCODE_64);

        env->ext_index--;

        if (env->ext_index == -1) {

            env->pending_int &= ~INTERRUPT_EXT_SERVICE;

        }

    } else {

        g_assert_not_reached();

    }



    mask = be64_to_cpu(lowcore->external_new_psw.mask);

    addr = be64_to_cpu(lowcore->external_new_psw.addr);

    lowcore->external_old_psw.mask = cpu_to_be64(get_psw_mask(env));

    lowcore->external_old_psw.addr = cpu_to_be64(env->psw.addr);



    cpu_unmap_lowcore(lowcore);



    DPRINTF("%s: %" PRIx64 " %" PRIx64 "\n", __func__,

            env->psw.mask, env->psw.addr);



    load_psw(env, mask, addr);

}
