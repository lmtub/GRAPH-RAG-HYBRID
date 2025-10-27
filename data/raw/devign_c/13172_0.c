void do_unassigned_access(target_phys_addr_t addr, int is_write, int is_exec,

                          int is_asi, int size)

{

    CPUState *saved_env;

    /* XXX: hack to restore env in all cases, even if not called from

       generated code */

    saved_env = env;

    env = cpu_single_env;

    qemu_log("Unassigned " TARGET_FMT_plx " wr=%d exe=%d\n",

             addr, is_write, is_exec);

    if (!(env->sregs[SR_MSR] & MSR_EE)) {

        return;

    }



    if (is_exec) {

        if (!(env->pvr.regs[2] & PVR2_IOPB_BUS_EXC_MASK)) {

            env->sregs[SR_ESR] = ESR_EC_INSN_BUS;

            helper_raise_exception(EXCP_HW_EXCP);

        }

    } else {

        if (!(env->pvr.regs[2] & PVR2_DOPB_BUS_EXC_MASK)) {

            env->sregs[SR_ESR] = ESR_EC_DATA_BUS;

            helper_raise_exception(EXCP_HW_EXCP);

        }

    }

}
