static void cpu_x86_dump_state(FILE *f)

{

    int eflags;

    char cc_op_name[32];

    eflags = cc_table[CC_OP].compute_all();

    eflags |= (DF & DF_MASK);

    if ((unsigned)env->cc_op < CC_OP_NB)

        strcpy(cc_op_name, cc_op_str[env->cc_op]);

    else

        snprintf(cc_op_name, sizeof(cc_op_name), "[%d]", env->cc_op);

    fprintf(f, 

            "EAX=%08x EBX=%08X ECX=%08x EDX=%08x\n"

            "ESI=%08x EDI=%08X EBP=%08x ESP=%08x\n"

            "CCS=%08x CCD=%08x CCO=%-8s EFL=%c%c%c%c%c%c%c\n"

            "EIP=%08x\n",

            env->regs[R_EAX], env->regs[R_EBX], env->regs[R_ECX], env->regs[R_EDX], 

            env->regs[R_ESI], env->regs[R_EDI], env->regs[R_EBP], env->regs[R_ESP], 

            env->cc_src, env->cc_dst, cc_op_name,

            eflags & DF_MASK ? 'D' : '-',

            eflags & CC_O ? 'O' : '-',

            eflags & CC_S ? 'S' : '-',

            eflags & CC_Z ? 'Z' : '-',

            eflags & CC_A ? 'A' : '-',

            eflags & CC_P ? 'P' : '-',

            eflags & CC_C ? 'C' : '-',

            env->eip);

#if 1

    fprintf(f, "ST0=%f ST1=%f ST2=%f ST3=%f\n", 

            (double)ST0, (double)ST1, (double)ST(2), (double)ST(3));

#endif

}
