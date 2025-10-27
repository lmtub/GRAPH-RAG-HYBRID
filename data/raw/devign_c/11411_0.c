do_kernel_trap(CPUARMState *env)

{

    uint32_t addr;

    uint32_t cpsr;

    uint32_t val;



    switch (env->regs[15]) {

    case 0xffff0fa0: /* __kernel_memory_barrier */

        /* ??? No-op. Will need to do better for SMP.  */

        break;

    case 0xffff0fc0: /* __kernel_cmpxchg */

         /* XXX: This only works between threads, not between processes.

            It's probably possible to implement this with native host

            operations. However things like ldrex/strex are much harder so

            there's not much point trying.  */

        start_exclusive();

        cpsr = cpsr_read(env);

        addr = env->regs[2];

        /* FIXME: This should SEGV if the access fails.  */

        if (get_user_u32(val, addr))

            val = ~env->regs[0];

        if (val == env->regs[0]) {

            val = env->regs[1];

            /* FIXME: Check for segfaults.  */

            put_user_u32(val, addr);

            env->regs[0] = 0;

            cpsr |= CPSR_C;

        } else {

            env->regs[0] = -1;

            cpsr &= ~CPSR_C;

        }

        cpsr_write(env, cpsr, CPSR_C);

        end_exclusive();

        break;

    case 0xffff0fe0: /* __kernel_get_tls */

        env->regs[0] = env->cp15.tpidrro_el0;

        break;

    case 0xffff0f60: /* __kernel_cmpxchg64 */

        arm_kernel_cmpxchg64_helper(env);

        break;



    default:

        return 1;

    }

    /* Jump back to the caller.  */

    addr = env->regs[14];

    if (addr & 1) {

        env->thumb = 1;

        addr &= ~1;

    }

    env->regs[15] = addr;



    return 0;

}
