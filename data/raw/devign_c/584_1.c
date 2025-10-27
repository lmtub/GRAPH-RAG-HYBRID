static inline void gen_intermediate_code_internal(X86CPU *cpu,

                                                  TranslationBlock *tb,

                                                  bool search_pc)

{

    CPUState *cs = CPU(cpu);

    CPUX86State *env = &cpu->env;

    DisasContext dc1, *dc = &dc1;

    target_ulong pc_ptr;

    uint16_t *gen_opc_end;

    CPUBreakpoint *bp;

    int j, lj;

    uint64_t flags;

    target_ulong pc_start;

    target_ulong cs_base;

    int num_insns;

    int max_insns;



    /* generate intermediate code */

    pc_start = tb->pc;

    cs_base = tb->cs_base;

    flags = tb->flags;



    dc->pe = (flags >> HF_PE_SHIFT) & 1;

    dc->code32 = (flags >> HF_CS32_SHIFT) & 1;

    dc->ss32 = (flags >> HF_SS32_SHIFT) & 1;

    dc->addseg = (flags >> HF_ADDSEG_SHIFT) & 1;

    dc->f_st = 0;

    dc->vm86 = (flags >> VM_SHIFT) & 1;

    dc->cpl = (flags >> HF_CPL_SHIFT) & 3;

    dc->iopl = (flags >> IOPL_SHIFT) & 3;

    dc->tf = (flags >> TF_SHIFT) & 1;

    dc->singlestep_enabled = cs->singlestep_enabled;

    dc->cc_op = CC_OP_DYNAMIC;

    dc->cc_op_dirty = false;

    dc->cs_base = cs_base;

    dc->tb = tb;

    dc->popl_esp_hack = 0;

    /* select memory access functions */

    dc->mem_index = 0;

    if (flags & HF_SOFTMMU_MASK) {

        dc->mem_index = cpu_mmu_index(env);

    }

    dc->cpuid_features = env->features[FEAT_1_EDX];

    dc->cpuid_ext_features = env->features[FEAT_1_ECX];

    dc->cpuid_ext2_features = env->features[FEAT_8000_0001_EDX];

    dc->cpuid_ext3_features = env->features[FEAT_8000_0001_ECX];

    dc->cpuid_7_0_ebx_features = env->features[FEAT_7_0_EBX];

#ifdef TARGET_X86_64

    dc->lma = (flags >> HF_LMA_SHIFT) & 1;

    dc->code64 = (flags >> HF_CS64_SHIFT) & 1;

#endif

    dc->flags = flags;

    dc->jmp_opt = !(dc->tf || cs->singlestep_enabled ||

                    (flags & HF_INHIBIT_IRQ_MASK)

#ifndef CONFIG_SOFTMMU

                    || (flags & HF_SOFTMMU_MASK)

#endif

                    );

#if 0

    /* check addseg logic */

    if (!dc->addseg && (dc->vm86 || !dc->pe || !dc->code32))

        printf("ERROR addseg\n");

#endif



    cpu_T[0] = tcg_temp_new();

    cpu_T[1] = tcg_temp_new();

    cpu_A0 = tcg_temp_new();



    cpu_tmp0 = tcg_temp_new();

    cpu_tmp1_i64 = tcg_temp_new_i64();

    cpu_tmp2_i32 = tcg_temp_new_i32();

    cpu_tmp3_i32 = tcg_temp_new_i32();

    cpu_tmp4 = tcg_temp_new();

    cpu_ptr0 = tcg_temp_new_ptr();

    cpu_ptr1 = tcg_temp_new_ptr();

    cpu_cc_srcT = tcg_temp_local_new();



    gen_opc_end = tcg_ctx.gen_opc_buf + OPC_MAX_SIZE;



    dc->is_jmp = DISAS_NEXT;

    pc_ptr = pc_start;

    lj = -1;

    num_insns = 0;

    max_insns = tb->cflags & CF_COUNT_MASK;

    if (max_insns == 0)

        max_insns = CF_COUNT_MASK;



    gen_tb_start();

    for(;;) {

        if (unlikely(!QTAILQ_EMPTY(&cs->breakpoints))) {

            QTAILQ_FOREACH(bp, &cs->breakpoints, entry) {

                if (bp->pc == pc_ptr &&

                    !((bp->flags & BP_CPU) && (tb->flags & HF_RF_MASK))) {

                    gen_debug(dc, pc_ptr - dc->cs_base);

                    break;

                }

            }

        }

        if (search_pc) {

            j = tcg_ctx.gen_opc_ptr - tcg_ctx.gen_opc_buf;

            if (lj < j) {

                lj++;

                while (lj < j)

                    tcg_ctx.gen_opc_instr_start[lj++] = 0;

            }

            tcg_ctx.gen_opc_pc[lj] = pc_ptr;

            gen_opc_cc_op[lj] = dc->cc_op;

            tcg_ctx.gen_opc_instr_start[lj] = 1;

            tcg_ctx.gen_opc_icount[lj] = num_insns;

        }

        if (num_insns + 1 == max_insns && (tb->cflags & CF_LAST_IO))

            gen_io_start();



        pc_ptr = disas_insn(env, dc, pc_ptr);

        num_insns++;

        /* stop translation if indicated */

        if (dc->is_jmp)

            break;

        /* if single step mode, we generate only one instruction and

           generate an exception */

        /* if irq were inhibited with HF_INHIBIT_IRQ_MASK, we clear

           the flag and abort the translation to give the irqs a

           change to be happen */

        if (dc->tf || dc->singlestep_enabled ||

            (flags & HF_INHIBIT_IRQ_MASK)) {

            gen_jmp_im(pc_ptr - dc->cs_base);

            gen_eob(dc);

            break;

        }

        /* if too long translation, stop generation too */

        if (tcg_ctx.gen_opc_ptr >= gen_opc_end ||

            (pc_ptr - pc_start) >= (TARGET_PAGE_SIZE - 32) ||

            num_insns >= max_insns) {

            gen_jmp_im(pc_ptr - dc->cs_base);

            gen_eob(dc);

            break;

        }

        if (singlestep) {

            gen_jmp_im(pc_ptr - dc->cs_base);

            gen_eob(dc);

            break;

        }

    }

    if (tb->cflags & CF_LAST_IO)

        gen_io_end();

    gen_tb_end(tb, num_insns);

    *tcg_ctx.gen_opc_ptr = INDEX_op_end;

    /* we don't forget to fill the last values */

    if (search_pc) {

        j = tcg_ctx.gen_opc_ptr - tcg_ctx.gen_opc_buf;

        lj++;

        while (lj <= j)

            tcg_ctx.gen_opc_instr_start[lj++] = 0;

    }



#ifdef DEBUG_DISAS

    if (qemu_loglevel_mask(CPU_LOG_TB_IN_ASM)) {

        int disas_flags;

        qemu_log("----------------\n");

        qemu_log("IN: %s\n", lookup_symbol(pc_start));

#ifdef TARGET_X86_64

        if (dc->code64)

            disas_flags = 2;

        else

#endif

            disas_flags = !dc->code32;

        log_target_disas(env, pc_start, pc_ptr - pc_start, disas_flags);

        qemu_log("\n");

    }

#endif



    if (!search_pc) {

        tb->size = pc_ptr - pc_start;

        tb->icount = num_insns;

    }

}
