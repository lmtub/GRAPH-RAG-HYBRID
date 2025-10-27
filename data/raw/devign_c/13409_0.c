static void dec_fpu(DisasContext *dc)

{

    if ((dc->tb_flags & MSR_EE_FLAG)

          && !(dc->env->pvr.regs[2] & PVR2_ILL_OPCODE_EXC_MASK)

          && !((dc->env->pvr.regs[2] & PVR2_USE_FPU_MASK))) {

        tcg_gen_movi_tl(cpu_SR[SR_ESR], ESR_EC_ILLEGAL_OP);

        t_gen_raise_exception(dc, EXCP_HW_EXCP);

        return;

    }



    qemu_log ("unimplemented FPU insn pc=%x opc=%x\n", dc->pc, dc->opcode);

    dc->abort_at_next_insn = 1;

}
