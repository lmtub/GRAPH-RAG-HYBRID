void helper_mtc0_status(CPUMIPSState *env, target_ulong arg1)

{

    MIPSCPU *cpu = mips_env_get_cpu(env);

    uint32_t val, old;

    uint32_t mask = env->CP0_Status_rw_bitmask;



    if (env->insn_flags & ISA_MIPS32R6) {

        if (extract32(env->CP0_Status, CP0St_KSU, 2) == 0x3) {

            mask &= ~(3 << CP0St_KSU);

        }

        mask &= ~(0x00180000 & arg1);

    }



    val = arg1 & mask;

    old = env->CP0_Status;

    env->CP0_Status = (env->CP0_Status & ~mask) | val;

    if (env->CP0_Config3 & (1 << CP0C3_MT)) {

        sync_c0_status(env, env, env->current_tc);

    } else {

        compute_hflags(env);

    }



    if (qemu_loglevel_mask(CPU_LOG_EXEC)) {

        qemu_log("Status %08x (%08x) => %08x (%08x) Cause %08x",

                old, old & env->CP0_Cause & CP0Ca_IP_mask,

                val, val & env->CP0_Cause & CP0Ca_IP_mask,

                env->CP0_Cause);

        switch (env->hflags & MIPS_HFLAG_KSU) {

        case MIPS_HFLAG_UM: qemu_log(", UM\n"); break;

        case MIPS_HFLAG_SM: qemu_log(", SM\n"); break;

        case MIPS_HFLAG_KM: qemu_log("\n"); break;

        default:

            cpu_abort(CPU(cpu), "Invalid MMU mode!\n");

            break;

        }

    }

}
