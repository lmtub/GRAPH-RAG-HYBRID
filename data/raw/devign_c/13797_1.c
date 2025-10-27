static target_ulong disas_insn(CPUX86State *env, DisasContext *s,
                               target_ulong pc_start)
{
    int b, prefixes, aflag, dflag;
    int shift, ot;
    int modrm, reg, rm, mod, reg_addr, op, opreg, offset_addr, val;
    target_ulong next_eip, tval;
    int rex_w, rex_r;
    if (unlikely(qemu_loglevel_mask(CPU_LOG_TB_OP | CPU_LOG_TB_OP_OPT))) {
        tcg_gen_debug_insn_start(pc_start);
    s->pc = pc_start;
    prefixes = 0;
    s->override = -1;
    rex_w = -1;
    rex_r = 0;
#ifdef TARGET_X86_64
    s->rex_x = 0;
    s->rex_b = 0;
    x86_64_hregs = 0;
#endif
    s->rip_offset = 0; /* for relative ip address */
    s->vex_l = 0;
    s->vex_v = 0;
 next_byte:
    b = cpu_ldub_code(env, s->pc);
    s->pc++;
    /* Collect prefixes.  */
    switch (b) {
    case 0xf3:
        prefixes |= PREFIX_REPZ;
        goto next_byte;
    case 0xf2:
        prefixes |= PREFIX_REPNZ;
        goto next_byte;
    case 0xf0:
        prefixes |= PREFIX_LOCK;
        goto next_byte;
    case 0x2e:
        s->override = R_CS;
        goto next_byte;
    case 0x36:
        s->override = R_SS;
        goto next_byte;
    case 0x3e:
        s->override = R_DS;
        goto next_byte;
    case 0x26:
        s->override = R_ES;
        goto next_byte;
    case 0x64:
        s->override = R_FS;
        goto next_byte;
    case 0x65:
        s->override = R_GS;
        goto next_byte;
    case 0x66:
        prefixes |= PREFIX_DATA;
        goto next_byte;
    case 0x67:
        prefixes |= PREFIX_ADR;
        goto next_byte;
#ifdef TARGET_X86_64
    case 0x40 ... 0x4f:
        if (CODE64(s)) {
            /* REX prefix */
            rex_w = (b >> 3) & 1;
            rex_r = (b & 0x4) << 1;
            s->rex_x = (b & 0x2) << 2;
            REX_B(s) = (b & 0x1) << 3;
            x86_64_hregs = 1; /* select uniform byte register addressing */
            goto next_byte;
        break;
#endif
    case 0xc5: /* 2-byte VEX */
    case 0xc4: /* 3-byte VEX */
        /* VEX prefixes cannot be used except in 32-bit mode.
           Otherwise the instruction is LES or LDS.  */
        if (s->code32 && !s->vm86) {
            static const int pp_prefix[4] = {
                0, PREFIX_DATA, PREFIX_REPZ, PREFIX_REPNZ
            };
            int vex3, vex2 = cpu_ldub_code(env, s->pc);
            if (!CODE64(s) && (vex2 & 0xc0) != 0xc0) {
                /* 4.1.4.6: In 32-bit mode, bits [7:6] must be 11b,
                   otherwise the instruction is LES or LDS.  */
                break;
            s->pc++;
            /* 4.1.1-4.1.3: No preceding lock, 66, f2, f3, or rex prefixes. */
            if (prefixes & (PREFIX_REPZ | PREFIX_REPNZ
                            | PREFIX_LOCK | PREFIX_DATA)) {
#ifdef TARGET_X86_64
            if (x86_64_hregs) {
#endif
            rex_r = (~vex2 >> 4) & 8;
            if (b == 0xc5) {
                vex3 = vex2;
                b = cpu_ldub_code(env, s->pc++);
            } else {
#ifdef TARGET_X86_64
                s->rex_x = (~vex2 >> 3) & 8;
                s->rex_b = (~vex2 >> 2) & 8;
#endif
                vex3 = cpu_ldub_code(env, s->pc++);
                rex_w = (vex3 >> 7) & 1;
                switch (vex2 & 0x1f) {
                case 0x01: /* Implied 0f leading opcode bytes.  */
                    b = cpu_ldub_code(env, s->pc++) | 0x100;
                    break;
                case 0x02: /* Implied 0f 38 leading opcode bytes.  */
                    b = 0x138;
                    break;
                case 0x03: /* Implied 0f 3a leading opcode bytes.  */
                    b = 0x13a;
                    break;
                default:   /* Reserved for future use.  */
            s->vex_v = (~vex3 >> 3) & 0xf;
            s->vex_l = (vex3 >> 2) & 1;
            prefixes |= pp_prefix[vex3 & 3] | PREFIX_VEX;
        break;
    /* Post-process prefixes.  */
    if (CODE64(s)) {
        /* In 64-bit mode, the default data size is 32-bit.  Select 64-bit
           data with rex_w, and 16-bit data with 0x66; rex_w takes precedence
           over 0x66 if both are present.  */
        dflag = (rex_w > 0 ? 2 : prefixes & PREFIX_DATA ? 0 : 1);
        /* In 64-bit mode, 0x67 selects 32-bit addressing.  */
        aflag = (prefixes & PREFIX_ADR ? 1 : 2);
    } else {
        /* In 16/32-bit mode, 0x66 selects the opposite data size.  */
        dflag = s->code32;
        if (prefixes & PREFIX_DATA) {
            dflag ^= 1;
        /* In 16/32-bit mode, 0x67 selects the opposite addressing.  */
        aflag = s->code32;
        if (prefixes & PREFIX_ADR) {
            aflag ^= 1;
    s->prefix = prefixes;
    s->aflag = aflag;
    s->dflag = dflag;
    /* lock generation */
    if (prefixes & PREFIX_LOCK)
        gen_helper_lock();
    /* now check op code */
 reswitch:
    switch(b) {
    case 0x0f:
        /**************************/
        /* extended op code */
        b = cpu_ldub_code(env, s->pc++) | 0x100;
        goto reswitch;
        /**************************/
        /* arith & logic */
    case 0x00 ... 0x05:
    case 0x08 ... 0x0d:
    case 0x10 ... 0x15:
    case 0x18 ... 0x1d:
    case 0x20 ... 0x25:
    case 0x28 ... 0x2d:
    case 0x30 ... 0x35:
    case 0x38 ... 0x3d:
        {
            int op, f, val;
            op = (b >> 3) & 7;
            f = (b >> 1) & 3;
            if ((b & 1) == 0)
                ot = OT_BYTE;
            else
                ot = dflag + OT_WORD;
            switch(f) {
            case 0: /* OP Ev, Gv */
                modrm = cpu_ldub_code(env, s->pc++);
                reg = ((modrm >> 3) & 7) | rex_r;
                mod = (modrm >> 6) & 3;
                rm = (modrm & 7) | REX_B(s);
                if (mod != 3) {
                    gen_lea_modrm(env, s, modrm, &reg_addr, &offset_addr);
                    opreg = OR_TMP0;
                } else if (op == OP_XORL && rm == reg) {
                xor_zero:
                    /* xor reg, reg optimisation */
                    set_cc_op(s, CC_OP_CLR);
                    gen_op_movl_T0_0();
                    gen_op_mov_reg_T0(ot, reg);
                    break;
                } else {
                    opreg = rm;
                gen_op_mov_TN_reg(ot, 1, reg);
                gen_op(s, op, ot, opreg);
                break;
            case 1: /* OP Gv, Ev */
                modrm = cpu_ldub_code(env, s->pc++);
                mod = (modrm >> 6) & 3;
                reg = ((modrm >> 3) & 7) | rex_r;
                rm = (modrm & 7) | REX_B(s);
                if (mod != 3) {
                    gen_lea_modrm(env, s, modrm, &reg_addr, &offset_addr);
                    gen_op_ld_T1_A0(ot + s->mem_index);
                } else if (op == OP_XORL && rm == reg) {
                    goto xor_zero;
                } else {
                    gen_op_mov_TN_reg(ot, 1, rm);
                gen_op(s, op, ot, reg);
                break;
            case 2: /* OP A, Iv */
                val = insn_get(env, s, ot);
                gen_op_movl_T1_im(val);
                gen_op(s, op, ot, OR_EAX);
                break;
        break;
    case 0x82:
        if (CODE64(s))
    case 0x80: /* GRP1 */
    case 0x81:
    case 0x83:
        {
            int val;
            if ((b & 1) == 0)
                ot = OT_BYTE;
            else
                ot = dflag + OT_WORD;
            modrm = cpu_ldub_code(env, s->pc++);
            mod = (modrm >> 6) & 3;
            rm = (modrm & 7) | REX_B(s);
            op = (modrm >> 3) & 7;
            if (mod != 3) {
                if (b == 0x83)
                    s->rip_offset = 1;
                else
                    s->rip_offset = insn_const_size(ot);
                gen_lea_modrm(env, s, modrm, &reg_addr, &offset_addr);
                opreg = OR_TMP0;
            } else {
                opreg = rm;
            switch(b) {
            default:
            case 0x80:
            case 0x81:
            case 0x82:
                val = insn_get(env, s, ot);
                break;
            case 0x83:
                val = (int8_t)insn_get(env, s, OT_BYTE);
                break;
            gen_op_movl_T1_im(val);
            gen_op(s, op, ot, opreg);
        break;
        /**************************/
        /* inc, dec, and other misc arith */
    case 0x40 ... 0x47: /* inc Gv */
        ot = dflag ? OT_LONG : OT_WORD;
        gen_inc(s, ot, OR_EAX + (b & 7), 1);
        break;
    case 0x48 ... 0x4f: /* dec Gv */
        ot = dflag ? OT_LONG : OT_WORD;
        gen_inc(s, ot, OR_EAX + (b & 7), -1);
        break;
    case 0xf6: /* GRP3 */
    case 0xf7:
        if ((b & 1) == 0)
            ot = OT_BYTE;
        else
            ot = dflag + OT_WORD;
        modrm = cpu_ldub_code(env, s->pc++);
        mod = (modrm >> 6) & 3;
        rm = (modrm & 7) | REX_B(s);
        op = (modrm >> 3) & 7;
        if (mod != 3) {
            if (op == 0)
                s->rip_offset = insn_const_size(ot);
            gen_lea_modrm(env, s, modrm, &reg_addr, &offset_addr);
            gen_op_ld_T0_A0(ot + s->mem_index);
        } else {
            gen_op_mov_TN_reg(ot, 0, rm);
        switch(op) {
        case 0: /* test */
            val = insn_get(env, s, ot);
            gen_op_movl_T1_im(val);
            gen_op_testl_T0_T1_cc();
            set_cc_op(s, CC_OP_LOGICB + ot);
            break;
        case 2: /* not */
            tcg_gen_not_tl(cpu_T[0], cpu_T[0]);
            if (mod != 3) {
                gen_op_st_T0_A0(ot + s->mem_index);
            } else {
                gen_op_mov_reg_T0(ot, rm);
            break;
        case 3: /* neg */
            tcg_gen_neg_tl(cpu_T[0], cpu_T[0]);
            if (mod != 3) {
                gen_op_st_T0_A0(ot + s->mem_index);
            } else {
                gen_op_mov_reg_T0(ot, rm);
            gen_op_update_neg_cc();
            set_cc_op(s, CC_OP_SUBB + ot);
            break;
        case 4: /* mul */
            switch(ot) {
            case OT_BYTE:
                gen_op_mov_TN_reg(OT_BYTE, 1, R_EAX);
                tcg_gen_ext8u_tl(cpu_T[0], cpu_T[0]);
                tcg_gen_ext8u_tl(cpu_T[1], cpu_T[1]);
                /* XXX: use 32 bit mul which could be faster */
                tcg_gen_mul_tl(cpu_T[0], cpu_T[0], cpu_T[1]);
                gen_op_mov_reg_T0(OT_WORD, R_EAX);
                tcg_gen_mov_tl(cpu_cc_dst, cpu_T[0]);
                tcg_gen_andi_tl(cpu_cc_src, cpu_T[0], 0xff00);
                set_cc_op(s, CC_OP_MULB);
                break;
            case OT_WORD:
                gen_op_mov_TN_reg(OT_WORD, 1, R_EAX);
                tcg_gen_ext16u_tl(cpu_T[0], cpu_T[0]);
                tcg_gen_ext16u_tl(cpu_T[1], cpu_T[1]);
                /* XXX: use 32 bit mul which could be faster */
                tcg_gen_mul_tl(cpu_T[0], cpu_T[0], cpu_T[1]);
                gen_op_mov_reg_T0(OT_WORD, R_EAX);
                tcg_gen_mov_tl(cpu_cc_dst, cpu_T[0]);
                tcg_gen_shri_tl(cpu_T[0], cpu_T[0], 16);
                gen_op_mov_reg_T0(OT_WORD, R_EDX);
                tcg_gen_mov_tl(cpu_cc_src, cpu_T[0]);
                set_cc_op(s, CC_OP_MULW);
                break;
            default:
            case OT_LONG:
                tcg_gen_trunc_tl_i32(cpu_tmp2_i32, cpu_T[0]);
                tcg_gen_trunc_tl_i32(cpu_tmp3_i32, cpu_regs[R_EAX]);
                tcg_gen_mulu2_i32(cpu_tmp2_i32, cpu_tmp3_i32,
                                  cpu_tmp2_i32, cpu_tmp3_i32);
                tcg_gen_extu_i32_tl(cpu_regs[R_EAX], cpu_tmp2_i32);
                tcg_gen_extu_i32_tl(cpu_regs[R_EDX], cpu_tmp3_i32);
                tcg_gen_mov_tl(cpu_cc_dst, cpu_regs[R_EAX]);
                tcg_gen_mov_tl(cpu_cc_src, cpu_regs[R_EDX]);
                set_cc_op(s, CC_OP_MULL);
                break;
#ifdef TARGET_X86_64
            case OT_QUAD:
                tcg_gen_mulu2_i64(cpu_regs[R_EAX], cpu_regs[R_EDX],
                                  cpu_T[0], cpu_regs[R_EAX]);
                tcg_gen_mov_tl(cpu_cc_dst, cpu_regs[R_EAX]);
                tcg_gen_mov_tl(cpu_cc_src, cpu_regs[R_EDX]);
                set_cc_op(s, CC_OP_MULQ);
                break;
#endif
            break;
        case 5: /* imul */
            switch(ot) {
            case OT_BYTE:
                gen_op_mov_TN_reg(OT_BYTE, 1, R_EAX);
                tcg_gen_ext8s_tl(cpu_T[0], cpu_T[0]);
                tcg_gen_ext8s_tl(cpu_T[1], cpu_T[1]);
                /* XXX: use 32 bit mul which could be faster */
                tcg_gen_mul_tl(cpu_T[0], cpu_T[0], cpu_T[1]);
                gen_op_mov_reg_T0(OT_WORD, R_EAX);
                tcg_gen_mov_tl(cpu_cc_dst, cpu_T[0]);
                tcg_gen_ext8s_tl(cpu_tmp0, cpu_T[0]);
                tcg_gen_sub_tl(cpu_cc_src, cpu_T[0], cpu_tmp0);
                set_cc_op(s, CC_OP_MULB);
                break;
            case OT_WORD:
                gen_op_mov_TN_reg(OT_WORD, 1, R_EAX);
                tcg_gen_ext16s_tl(cpu_T[0], cpu_T[0]);
                tcg_gen_ext16s_tl(cpu_T[1], cpu_T[1]);
                /* XXX: use 32 bit mul which could be faster */
                tcg_gen_mul_tl(cpu_T[0], cpu_T[0], cpu_T[1]);
                gen_op_mov_reg_T0(OT_WORD, R_EAX);
                tcg_gen_mov_tl(cpu_cc_dst, cpu_T[0]);
                tcg_gen_ext16s_tl(cpu_tmp0, cpu_T[0]);
                tcg_gen_sub_tl(cpu_cc_src, cpu_T[0], cpu_tmp0);
                tcg_gen_shri_tl(cpu_T[0], cpu_T[0], 16);
                gen_op_mov_reg_T0(OT_WORD, R_EDX);
                set_cc_op(s, CC_OP_MULW);
                break;
            default:
            case OT_LONG:
                tcg_gen_trunc_tl_i32(cpu_tmp2_i32, cpu_T[0]);
                tcg_gen_trunc_tl_i32(cpu_tmp3_i32, cpu_regs[R_EAX]);
                tcg_gen_muls2_i32(cpu_tmp2_i32, cpu_tmp3_i32,
                                  cpu_tmp2_i32, cpu_tmp3_i32);
                tcg_gen_extu_i32_tl(cpu_regs[R_EAX], cpu_tmp2_i32);
                tcg_gen_extu_i32_tl(cpu_regs[R_EDX], cpu_tmp3_i32);
                tcg_gen_sari_i32(cpu_tmp2_i32, cpu_tmp2_i32, 31);
                tcg_gen_mov_tl(cpu_cc_dst, cpu_regs[R_EAX]);
                tcg_gen_sub_i32(cpu_tmp2_i32, cpu_tmp2_i32, cpu_tmp3_i32);
                tcg_gen_extu_i32_tl(cpu_cc_src, cpu_tmp2_i32);
                set_cc_op(s, CC_OP_MULL);
                break;
#ifdef TARGET_X86_64
            case OT_QUAD:
                tcg_gen_muls2_i64(cpu_regs[R_EAX], cpu_regs[R_EDX],
                                  cpu_T[0], cpu_regs[R_EAX]);
                tcg_gen_mov_tl(cpu_cc_dst, cpu_regs[R_EAX]);
                tcg_gen_sari_tl(cpu_cc_src, cpu_regs[R_EAX], 63);
                tcg_gen_sub_tl(cpu_cc_src, cpu_cc_src, cpu_regs[R_EDX]);
                set_cc_op(s, CC_OP_MULQ);
                break;
#endif
            break;
        case 6: /* div */
            switch(ot) {
            case OT_BYTE:
                gen_jmp_im(pc_start - s->cs_base);
                gen_helper_divb_AL(cpu_env, cpu_T[0]);
                break;
            case OT_WORD:
                gen_jmp_im(pc_start - s->cs_base);
                gen_helper_divw_AX(cpu_env, cpu_T[0]);
                break;
            default:
            case OT_LONG:
                gen_jmp_im(pc_start - s->cs_base);
                gen_helper_divl_EAX(cpu_env, cpu_T[0]);
                break;
#ifdef TARGET_X86_64
            case OT_QUAD:
                gen_jmp_im(pc_start - s->cs_base);
                gen_helper_divq_EAX(cpu_env, cpu_T[0]);
                break;
#endif
            break;
        case 7: /* idiv */
            switch(ot) {
            case OT_BYTE:
                gen_jmp_im(pc_start - s->cs_base);
                gen_helper_idivb_AL(cpu_env, cpu_T[0]);
                break;
            case OT_WORD:
                gen_jmp_im(pc_start - s->cs_base);
                gen_helper_idivw_AX(cpu_env, cpu_T[0]);
                break;
            default:
            case OT_LONG:
                gen_jmp_im(pc_start - s->cs_base);
                gen_helper_idivl_EAX(cpu_env, cpu_T[0]);
                break;
#ifdef TARGET_X86_64
            case OT_QUAD:
                gen_jmp_im(pc_start - s->cs_base);
                gen_helper_idivq_EAX(cpu_env, cpu_T[0]);
                break;
#endif
            break;
        default:
        break;
    case 0xfe: /* GRP4 */
    case 0xff: /* GRP5 */
        if ((b & 1) == 0)
            ot = OT_BYTE;
        else
            ot = dflag + OT_WORD;
        modrm = cpu_ldub_code(env, s->pc++);
        mod = (modrm >> 6) & 3;
        rm = (modrm & 7) | REX_B(s);
        op = (modrm >> 3) & 7;
        if (op >= 2 && b == 0xfe) {
        if (CODE64(s)) {
            if (op == 2 || op == 4) {
                /* operand size for jumps is 64 bit */
                ot = OT_QUAD;
            } else if (op == 3 || op == 5) {
                ot = dflag ? OT_LONG + (rex_w == 1) : OT_WORD;
            } else if (op == 6) {
                /* default push size is 64 bit */
                ot = dflag ? OT_QUAD : OT_WORD;
        if (mod != 3) {
            gen_lea_modrm(env, s, modrm, &reg_addr, &offset_addr);
            if (op >= 2 && op != 3 && op != 5)
                gen_op_ld_T0_A0(ot + s->mem_index);
        } else {
            gen_op_mov_TN_reg(ot, 0, rm);
        switch(op) {
        case 0: /* inc Ev */
            if (mod != 3)
                opreg = OR_TMP0;
            else
                opreg = rm;
            gen_inc(s, ot, opreg, 1);
            break;
        case 1: /* dec Ev */
            if (mod != 3)
                opreg = OR_TMP0;
            else
                opreg = rm;
            gen_inc(s, ot, opreg, -1);
            break;
        case 2: /* call Ev */
            /* XXX: optimize if memory (no 'and' is necessary) */
            if (s->dflag == 0)
                gen_op_andl_T0_ffff();
            next_eip = s->pc - s->cs_base;
            gen_movtl_T1_im(next_eip);
            gen_push_T1(s);
            gen_op_jmp_T0();
            gen_eob(s);
            break;
        case 3: /* lcall Ev */
            gen_op_ld_T1_A0(ot + s->mem_index);
            gen_add_A0_im(s, 1 << (ot - OT_WORD + 1));
            gen_op_ldu_T0_A0(OT_WORD + s->mem_index);
        do_lcall:
            if (s->pe && !s->vm86) {
                gen_update_cc_op(s);
                gen_jmp_im(pc_start - s->cs_base);
                tcg_gen_trunc_tl_i32(cpu_tmp2_i32, cpu_T[0]);
                gen_helper_lcall_protected(cpu_env, cpu_tmp2_i32, cpu_T[1],
                                           tcg_const_i32(dflag),
                                           tcg_const_i32(s->pc - pc_start));
            } else {
                tcg_gen_trunc_tl_i32(cpu_tmp2_i32, cpu_T[0]);
                gen_helper_lcall_real(cpu_env, cpu_tmp2_i32, cpu_T[1],
                                      tcg_const_i32(dflag),
                                      tcg_const_i32(s->pc - s->cs_base));
            gen_eob(s);
            break;
        case 4: /* jmp Ev */
            if (s->dflag == 0)
                gen_op_andl_T0_ffff();
            gen_op_jmp_T0();
            gen_eob(s);
            break;
        case 5: /* ljmp Ev */
            gen_op_ld_T1_A0(ot + s->mem_index);
            gen_add_A0_im(s, 1 << (ot - OT_WORD + 1));
            gen_op_ldu_T0_A0(OT_WORD + s->mem_index);
        do_ljmp:
            if (s->pe && !s->vm86) {
                gen_update_cc_op(s);
                gen_jmp_im(pc_start - s->cs_base);
                tcg_gen_trunc_tl_i32(cpu_tmp2_i32, cpu_T[0]);
                gen_helper_ljmp_protected(cpu_env, cpu_tmp2_i32, cpu_T[1],
                                          tcg_const_i32(s->pc - pc_start));
            } else {
                gen_op_movl_seg_T0_vm(R_CS);
                gen_op_movl_T0_T1();
                gen_op_jmp_T0();
            gen_eob(s);
            break;
        case 6: /* push Ev */
            gen_push_T0(s);
            break;
        default:
        break;
    case 0x84: /* test Ev, Gv */
    case 0x85:
        if ((b & 1) == 0)
            ot = OT_BYTE;
        else
            ot = dflag + OT_WORD;
        modrm = cpu_ldub_code(env, s->pc++);
        reg = ((modrm >> 3) & 7) | rex_r;
        gen_ldst_modrm(env, s, modrm, ot, OR_TMP0, 0);
        gen_op_mov_TN_reg(ot, 1, reg);
        gen_op_testl_T0_T1_cc();
        set_cc_op(s, CC_OP_LOGICB + ot);
        break;
    case 0xa8: /* test eAX, Iv */
    case 0xa9:
        if ((b & 1) == 0)
            ot = OT_BYTE;
        else
            ot = dflag + OT_WORD;
        val = insn_get(env, s, ot);
        gen_op_mov_TN_reg(ot, 0, OR_EAX);
        gen_op_movl_T1_im(val);
        gen_op_testl_T0_T1_cc();
        set_cc_op(s, CC_OP_LOGICB + ot);
        break;
    case 0x98: /* CWDE/CBW */
#ifdef TARGET_X86_64
        if (dflag == 2) {
            gen_op_mov_TN_reg(OT_LONG, 0, R_EAX);
            tcg_gen_ext32s_tl(cpu_T[0], cpu_T[0]);
            gen_op_mov_reg_T0(OT_QUAD, R_EAX);
        } else
#endif
        if (dflag == 1) {
            gen_op_mov_TN_reg(OT_WORD, 0, R_EAX);
            tcg_gen_ext16s_tl(cpu_T[0], cpu_T[0]);
            gen_op_mov_reg_T0(OT_LONG, R_EAX);
        } else {
            gen_op_mov_TN_reg(OT_BYTE, 0, R_EAX);
            tcg_gen_ext8s_tl(cpu_T[0], cpu_T[0]);
            gen_op_mov_reg_T0(OT_WORD, R_EAX);
        break;
    case 0x99: /* CDQ/CWD */
#ifdef TARGET_X86_64
        if (dflag == 2) {
            gen_op_mov_TN_reg(OT_QUAD, 0, R_EAX);
            tcg_gen_sari_tl(cpu_T[0], cpu_T[0], 63);
            gen_op_mov_reg_T0(OT_QUAD, R_EDX);
        } else
#endif
        if (dflag == 1) {
            gen_op_mov_TN_reg(OT_LONG, 0, R_EAX);
            tcg_gen_ext32s_tl(cpu_T[0], cpu_T[0]);
            tcg_gen_sari_tl(cpu_T[0], cpu_T[0], 31);
            gen_op_mov_reg_T0(OT_LONG, R_EDX);
        } else {
            gen_op_mov_TN_reg(OT_WORD, 0, R_EAX);
            tcg_gen_ext16s_tl(cpu_T[0], cpu_T[0]);
            tcg_gen_sari_tl(cpu_T[0], cpu_T[0], 15);
            gen_op_mov_reg_T0(OT_WORD, R_EDX);
        break;
    case 0x1af: /* imul Gv, Ev */
    case 0x69: /* imul Gv, Ev, I */
    case 0x6b:
        ot = dflag + OT_WORD;
        modrm = cpu_ldub_code(env, s->pc++);
        reg = ((modrm >> 3) & 7) | rex_r;
        if (b == 0x69)
            s->rip_offset = insn_const_size(ot);
        else if (b == 0x6b)
            s->rip_offset = 1;
        gen_ldst_modrm(env, s, modrm, ot, OR_TMP0, 0);
        if (b == 0x69) {
            val = insn_get(env, s, ot);
            gen_op_movl_T1_im(val);
        } else if (b == 0x6b) {
            val = (int8_t)insn_get(env, s, OT_BYTE);
            gen_op_movl_T1_im(val);
        } else {
            gen_op_mov_TN_reg(ot, 1, reg);
        switch (ot) {
#ifdef TARGET_X86_64
        case OT_QUAD:
            tcg_gen_muls2_i64(cpu_regs[reg], cpu_T[1], cpu_T[0], cpu_T[1]);
            tcg_gen_mov_tl(cpu_cc_dst, cpu_regs[reg]);
            tcg_gen_sari_tl(cpu_cc_src, cpu_cc_dst, 63);
            tcg_gen_sub_tl(cpu_cc_src, cpu_cc_src, cpu_T[1]);
            break;
#endif
        case OT_LONG:
            tcg_gen_trunc_tl_i32(cpu_tmp2_i32, cpu_T[0]);
            tcg_gen_trunc_tl_i32(cpu_tmp3_i32, cpu_T[1]);
            tcg_gen_muls2_i32(cpu_tmp2_i32, cpu_tmp3_i32,
                              cpu_tmp2_i32, cpu_tmp3_i32);
            tcg_gen_extu_i32_tl(cpu_regs[reg], cpu_tmp2_i32);
            tcg_gen_sari_i32(cpu_tmp2_i32, cpu_tmp2_i32, 31);
            tcg_gen_mov_tl(cpu_cc_dst, cpu_regs[reg]);
            tcg_gen_sub_i32(cpu_tmp2_i32, cpu_tmp2_i32, cpu_tmp3_i32);
            tcg_gen_extu_i32_tl(cpu_cc_src, cpu_tmp2_i32);
            break;
        default:
            tcg_gen_ext16s_tl(cpu_T[0], cpu_T[0]);
            tcg_gen_ext16s_tl(cpu_T[1], cpu_T[1]);
            /* XXX: use 32 bit mul which could be faster */
            tcg_gen_mul_tl(cpu_T[0], cpu_T[0], cpu_T[1]);
            tcg_gen_mov_tl(cpu_cc_dst, cpu_T[0]);
            tcg_gen_ext16s_tl(cpu_tmp0, cpu_T[0]);
            tcg_gen_sub_tl(cpu_cc_src, cpu_T[0], cpu_tmp0);
            gen_op_mov_reg_T0(ot, reg);
            break;
        set_cc_op(s, CC_OP_MULB + ot);
        break;
    case 0x1c0:
    case 0x1c1: /* xadd Ev, Gv */
        if ((b & 1) == 0)
            ot = OT_BYTE;
        else
            ot = dflag + OT_WORD;
        modrm = cpu_ldub_code(env, s->pc++);
        reg = ((modrm >> 3) & 7) | rex_r;
        mod = (modrm >> 6) & 3;
        if (mod == 3) {
            rm = (modrm & 7) | REX_B(s);
            gen_op_mov_TN_reg(ot, 0, reg);
            gen_op_mov_TN_reg(ot, 1, rm);
            gen_op_addl_T0_T1();
            gen_op_mov_reg_T1(ot, reg);
            gen_op_mov_reg_T0(ot, rm);
        } else {
            gen_lea_modrm(env, s, modrm, &reg_addr, &offset_addr);
            gen_op_mov_TN_reg(ot, 0, reg);
            gen_op_ld_T1_A0(ot + s->mem_index);
            gen_op_addl_T0_T1();
            gen_op_st_T0_A0(ot + s->mem_index);
            gen_op_mov_reg_T1(ot, reg);
        gen_op_update2_cc();
        set_cc_op(s, CC_OP_ADDB + ot);
        break;
    case 0x1b0:
    case 0x1b1: /* cmpxchg Ev, Gv */
        {
            int label1, label2;
            TCGv t0, t1, t2, a0;
            if ((b & 1) == 0)
                ot = OT_BYTE;
            else
                ot = dflag + OT_WORD;
            modrm = cpu_ldub_code(env, s->pc++);
            reg = ((modrm >> 3) & 7) | rex_r;
            mod = (modrm >> 6) & 3;
            t0 = tcg_temp_local_new();
            t1 = tcg_temp_local_new();
            t2 = tcg_temp_local_new();
            a0 = tcg_temp_local_new();
            gen_op_mov_v_reg(ot, t1, reg);
            if (mod == 3) {
                rm = (modrm & 7) | REX_B(s);
                gen_op_mov_v_reg(ot, t0, rm);
            } else {
                gen_lea_modrm(env, s, modrm, &reg_addr, &offset_addr);
                tcg_gen_mov_tl(a0, cpu_A0);
                gen_op_ld_v(ot + s->mem_index, t0, a0);
                rm = 0; /* avoid warning */
            label1 = gen_new_label();
            tcg_gen_mov_tl(t2, cpu_regs[R_EAX]);
            gen_extu(ot, t0);
            gen_extu(ot, t2);
            tcg_gen_brcond_tl(TCG_COND_EQ, t2, t0, label1);
            label2 = gen_new_label();
            if (mod == 3) {
                gen_op_mov_reg_v(ot, R_EAX, t0);
                tcg_gen_br(label2);
                gen_set_label(label1);
                gen_op_mov_reg_v(ot, rm, t1);
            } else {
                /* perform no-op store cycle like physical cpu; must be
                   before changing accumulator to ensure idempotency if
                   the store faults and the instruction is restarted */
                gen_op_st_v(ot + s->mem_index, t0, a0);
                gen_op_mov_reg_v(ot, R_EAX, t0);
                tcg_gen_br(label2);
                gen_set_label(label1);
                gen_op_st_v(ot + s->mem_index, t1, a0);
            gen_set_label(label2);
            tcg_gen_mov_tl(cpu_cc_src, t0);
            tcg_gen_mov_tl(cpu_cc_srcT, t2);
            tcg_gen_sub_tl(cpu_cc_dst, t2, t0);
            set_cc_op(s, CC_OP_SUBB + ot);
            tcg_temp_free(t0);
            tcg_temp_free(t1);
            tcg_temp_free(t2);
            tcg_temp_free(a0);
        break;
    case 0x1c7: /* cmpxchg8b */
        modrm = cpu_ldub_code(env, s->pc++);
        mod = (modrm >> 6) & 3;
        if ((mod == 3) || ((modrm & 0x38) != 0x8))
#ifdef TARGET_X86_64
        if (dflag == 2) {
            if (!(s->cpuid_ext_features & CPUID_EXT_CX16))
            gen_jmp_im(pc_start - s->cs_base);
            gen_update_cc_op(s);
            gen_lea_modrm(env, s, modrm, &reg_addr, &offset_addr);
            gen_helper_cmpxchg16b(cpu_env, cpu_A0);
        } else
#endif        
        {
            if (!(s->cpuid_features & CPUID_CX8))
            gen_jmp_im(pc_start - s->cs_base);
            gen_update_cc_op(s);
            gen_lea_modrm(env, s, modrm, &reg_addr, &offset_addr);
            gen_helper_cmpxchg8b(cpu_env, cpu_A0);
        set_cc_op(s, CC_OP_EFLAGS);
        break;
        /**************************/
        /* push/pop */
    case 0x50 ... 0x57: /* push */
        gen_op_mov_TN_reg(OT_LONG, 0, (b & 7) | REX_B(s));
        gen_push_T0(s);
        break;
    case 0x58 ... 0x5f: /* pop */
        if (CODE64(s)) {
            ot = dflag ? OT_QUAD : OT_WORD;
        } else {
            ot = dflag + OT_WORD;
        gen_pop_T0(s);
        /* NOTE: order is important for pop %sp */
        gen_pop_update(s);
        gen_op_mov_reg_T0(ot, (b & 7) | REX_B(s));
        break;
    case 0x60: /* pusha */
        if (CODE64(s))
        gen_pusha(s);
        break;
    case 0x61: /* popa */
        if (CODE64(s))
        gen_popa(s);
        break;
    case 0x68: /* push Iv */
    case 0x6a:
        if (CODE64(s)) {
            ot = dflag ? OT_QUAD : OT_WORD;
        } else {
            ot = dflag + OT_WORD;
        if (b == 0x68)
            val = insn_get(env, s, ot);
        else
            val = (int8_t)insn_get(env, s, OT_BYTE);
        gen_op_movl_T0_im(val);
        gen_push_T0(s);
        break;
    case 0x8f: /* pop Ev */
        if (CODE64(s)) {
            ot = dflag ? OT_QUAD : OT_WORD;
        } else {
            ot = dflag + OT_WORD;
        modrm = cpu_ldub_code(env, s->pc++);
        mod = (modrm >> 6) & 3;
        gen_pop_T0(s);
        if (mod == 3) {
            /* NOTE: order is important for pop %sp */
            gen_pop_update(s);
            rm = (modrm & 7) | REX_B(s);
            gen_op_mov_reg_T0(ot, rm);
        } else {
            /* NOTE: order is important too for MMU exceptions */
            s->popl_esp_hack = 1 << ot;
            gen_ldst_modrm(env, s, modrm, ot, OR_TMP0, 1);
            s->popl_esp_hack = 0;
            gen_pop_update(s);
        break;
    case 0xc8: /* enter */
        {
            int level;
            val = cpu_lduw_code(env, s->pc);
            s->pc += 2;
            level = cpu_ldub_code(env, s->pc++);
            gen_enter(s, val, level);
        break;
    case 0xc9: /* leave */
        /* XXX: exception not precise (ESP is updated before potential exception) */
        if (CODE64(s)) {
            gen_op_mov_TN_reg(OT_QUAD, 0, R_EBP);
            gen_op_mov_reg_T0(OT_QUAD, R_ESP);
        } else if (s->ss32) {
            gen_op_mov_TN_reg(OT_LONG, 0, R_EBP);
            gen_op_mov_reg_T0(OT_LONG, R_ESP);
        } else {
            gen_op_mov_TN_reg(OT_WORD, 0, R_EBP);
            gen_op_mov_reg_T0(OT_WORD, R_ESP);
        gen_pop_T0(s);
        if (CODE64(s)) {
            ot = dflag ? OT_QUAD : OT_WORD;
        } else {
            ot = dflag + OT_WORD;
        gen_op_mov_reg_T0(ot, R_EBP);
        gen_pop_update(s);
        break;
    case 0x06: /* push es */
    case 0x0e: /* push cs */
    case 0x16: /* push ss */
    case 0x1e: /* push ds */
        if (CODE64(s))
        gen_op_movl_T0_seg(b >> 3);
        gen_push_T0(s);
        break;
    case 0x1a0: /* push fs */
    case 0x1a8: /* push gs */
        gen_op_movl_T0_seg((b >> 3) & 7);
        gen_push_T0(s);
        break;
    case 0x07: /* pop es */
    case 0x17: /* pop ss */
    case 0x1f: /* pop ds */
        if (CODE64(s))
        reg = b >> 3;
        gen_pop_T0(s);
        gen_movl_seg_T0(s, reg, pc_start - s->cs_base);
        gen_pop_update(s);
        if (reg == R_SS) {
            /* if reg == SS, inhibit interrupts/trace. */
            /* If several instructions disable interrupts, only the
               _first_ does it */
            if (!(s->tb->flags & HF_INHIBIT_IRQ_MASK))
                gen_helper_set_inhibit_irq(cpu_env);
            s->tf = 0;
        if (s->is_jmp) {
            gen_jmp_im(s->pc - s->cs_base);
            gen_eob(s);
        break;
    case 0x1a1: /* pop fs */
    case 0x1a9: /* pop gs */
        gen_pop_T0(s);
        gen_movl_seg_T0(s, (b >> 3) & 7, pc_start - s->cs_base);
        gen_pop_update(s);
        if (s->is_jmp) {
            gen_jmp_im(s->pc - s->cs_base);
            gen_eob(s);
        break;
        /**************************/
        /* mov */
    case 0x88:
    case 0x89: /* mov Gv, Ev */
        if ((b & 1) == 0)
            ot = OT_BYTE;
        else
            ot = dflag + OT_WORD;
        modrm = cpu_ldub_code(env, s->pc++);
        reg = ((modrm >> 3) & 7) | rex_r;
        /* generate a generic store */
        gen_ldst_modrm(env, s, modrm, ot, reg, 1);
        break;
    case 0xc6:
    case 0xc7: /* mov Ev, Iv */
        if ((b & 1) == 0)
            ot = OT_BYTE;
        else
            ot = dflag + OT_WORD;
        modrm = cpu_ldub_code(env, s->pc++);
        mod = (modrm >> 6) & 3;
        if (mod != 3) {
            s->rip_offset = insn_const_size(ot);
            gen_lea_modrm(env, s, modrm, &reg_addr, &offset_addr);
        val = insn_get(env, s, ot);
        gen_op_movl_T0_im(val);
        if (mod != 3)
            gen_op_st_T0_A0(ot + s->mem_index);
        else
            gen_op_mov_reg_T0(ot, (modrm & 7) | REX_B(s));
        break;
    case 0x8a:
    case 0x8b: /* mov Ev, Gv */
        if ((b & 1) == 0)
            ot = OT_BYTE;
        else
            ot = OT_WORD + dflag;
        modrm = cpu_ldub_code(env, s->pc++);
        reg = ((modrm >> 3) & 7) | rex_r;
        gen_ldst_modrm(env, s, modrm, ot, OR_TMP0, 0);
        gen_op_mov_reg_T0(ot, reg);
        break;
    case 0x8e: /* mov seg, Gv */
        modrm = cpu_ldub_code(env, s->pc++);
        reg = (modrm >> 3) & 7;
        if (reg >= 6 || reg == R_CS)
        gen_ldst_modrm(env, s, modrm, OT_WORD, OR_TMP0, 0);
        gen_movl_seg_T0(s, reg, pc_start - s->cs_base);
        if (reg == R_SS) {
            /* if reg == SS, inhibit interrupts/trace */
            /* If several instructions disable interrupts, only the
               _first_ does it */
            if (!(s->tb->flags & HF_INHIBIT_IRQ_MASK))
                gen_helper_set_inhibit_irq(cpu_env);
            s->tf = 0;
        if (s->is_jmp) {
            gen_jmp_im(s->pc - s->cs_base);
            gen_eob(s);
        break;
    case 0x8c: /* mov Gv, seg */
        modrm = cpu_ldub_code(env, s->pc++);
        reg = (modrm >> 3) & 7;
        mod = (modrm >> 6) & 3;
        if (reg >= 6)
        gen_op_movl_T0_seg(reg);
        if (mod == 3)
            ot = OT_WORD + dflag;
        else
            ot = OT_WORD;
        gen_ldst_modrm(env, s, modrm, ot, OR_TMP0, 1);
        break;
    case 0x1b6: /* movzbS Gv, Eb */
    case 0x1b7: /* movzwS Gv, Eb */
    case 0x1be: /* movsbS Gv, Eb */
    case 0x1bf: /* movswS Gv, Eb */
        {
            int d_ot;
            /* d_ot is the size of destination */
            d_ot = dflag + OT_WORD;
            /* ot is the size of source */
            ot = (b & 1) + OT_BYTE;
            modrm = cpu_ldub_code(env, s->pc++);
            reg = ((modrm >> 3) & 7) | rex_r;
            mod = (modrm >> 6) & 3;
            rm = (modrm & 7) | REX_B(s);
            if (mod == 3) {
                gen_op_mov_TN_reg(ot, 0, rm);
                switch(ot | (b & 8)) {
                case OT_BYTE:
                    tcg_gen_ext8u_tl(cpu_T[0], cpu_T[0]);
                    break;
                case OT_BYTE | 8:
                    tcg_gen_ext8s_tl(cpu_T[0], cpu_T[0]);
                    break;
                case OT_WORD:
                    tcg_gen_ext16u_tl(cpu_T[0], cpu_T[0]);
                    break;
                default:
                case OT_WORD | 8:
                    tcg_gen_ext16s_tl(cpu_T[0], cpu_T[0]);
                    break;
                gen_op_mov_reg_T0(d_ot, reg);
            } else {
                gen_lea_modrm(env, s, modrm, &reg_addr, &offset_addr);
                if (b & 8) {
                    gen_op_lds_T0_A0(ot + s->mem_index);
                } else {
                    gen_op_ldu_T0_A0(ot + s->mem_index);
                gen_op_mov_reg_T0(d_ot, reg);
        break;
    case 0x8d: /* lea */
        ot = dflag + OT_WORD;
        modrm = cpu_ldub_code(env, s->pc++);
        mod = (modrm >> 6) & 3;
        if (mod == 3)
        reg = ((modrm >> 3) & 7) | rex_r;
        /* we must ensure that no segment is added */
        s->override = -1;
        val = s->addseg;
        s->addseg = 0;
        gen_lea_modrm(env, s, modrm, &reg_addr, &offset_addr);
        s->addseg = val;
        gen_op_mov_reg_A0(ot - OT_WORD, reg);
        break;
    case 0xa0: /* mov EAX, Ov */
    case 0xa1:
    case 0xa2: /* mov Ov, EAX */
    case 0xa3:
        {
            target_ulong offset_addr;
            if ((b & 1) == 0)
                ot = OT_BYTE;
            else
                ot = dflag + OT_WORD;
#ifdef TARGET_X86_64
            if (s->aflag == 2) {
                offset_addr = cpu_ldq_code(env, s->pc);
                s->pc += 8;
                gen_op_movq_A0_im(offset_addr);
            } else
#endif
            {
                if (s->aflag) {
                    offset_addr = insn_get(env, s, OT_LONG);
                } else {
                    offset_addr = insn_get(env, s, OT_WORD);
                gen_op_movl_A0_im(offset_addr);
            gen_add_A0_ds_seg(s);
            if ((b & 2) == 0) {
                gen_op_ld_T0_A0(ot + s->mem_index);
                gen_op_mov_reg_T0(ot, R_EAX);
            } else {
                gen_op_mov_TN_reg(ot, 0, R_EAX);
                gen_op_st_T0_A0(ot + s->mem_index);
        break;
    case 0xd7: /* xlat */
#ifdef TARGET_X86_64
        if (s->aflag == 2) {
            gen_op_movq_A0_reg(R_EBX);
            gen_op_mov_TN_reg(OT_QUAD, 0, R_EAX);
            tcg_gen_andi_tl(cpu_T[0], cpu_T[0], 0xff);
            tcg_gen_add_tl(cpu_A0, cpu_A0, cpu_T[0]);
        } else
#endif
        {
            gen_op_movl_A0_reg(R_EBX);
            gen_op_mov_TN_reg(OT_LONG, 0, R_EAX);
            tcg_gen_andi_tl(cpu_T[0], cpu_T[0], 0xff);
            tcg_gen_add_tl(cpu_A0, cpu_A0, cpu_T[0]);
            if (s->aflag == 0)
                gen_op_andl_A0_ffff();
            else
                tcg_gen_andi_tl(cpu_A0, cpu_A0, 0xffffffff);
        gen_add_A0_ds_seg(s);
        gen_op_ldu_T0_A0(OT_BYTE + s->mem_index);
        gen_op_mov_reg_T0(OT_BYTE, R_EAX);
        break;
    case 0xb0 ... 0xb7: /* mov R, Ib */
        val = insn_get(env, s, OT_BYTE);
        gen_op_movl_T0_im(val);
        gen_op_mov_reg_T0(OT_BYTE, (b & 7) | REX_B(s));
        break;
    case 0xb8 ... 0xbf: /* mov R, Iv */
#ifdef TARGET_X86_64
        if (dflag == 2) {
            uint64_t tmp;
            /* 64 bit case */
            tmp = cpu_ldq_code(env, s->pc);
            s->pc += 8;
            reg = (b & 7) | REX_B(s);
            gen_movtl_T0_im(tmp);
            gen_op_mov_reg_T0(OT_QUAD, reg);
        } else
#endif
        {
            ot = dflag ? OT_LONG : OT_WORD;
            val = insn_get(env, s, ot);
            reg = (b & 7) | REX_B(s);
            gen_op_movl_T0_im(val);
            gen_op_mov_reg_T0(ot, reg);
        break;
    case 0x91 ... 0x97: /* xchg R, EAX */
    do_xchg_reg_eax:
        ot = dflag + OT_WORD;
        reg = (b & 7) | REX_B(s);
        rm = R_EAX;
        goto do_xchg_reg;
    case 0x86:
    case 0x87: /* xchg Ev, Gv */
        if ((b & 1) == 0)
            ot = OT_BYTE;
        else
            ot = dflag + OT_WORD;
        modrm = cpu_ldub_code(env, s->pc++);
        reg = ((modrm >> 3) & 7) | rex_r;
        mod = (modrm >> 6) & 3;
        if (mod == 3) {
            rm = (modrm & 7) | REX_B(s);
        do_xchg_reg:
            gen_op_mov_TN_reg(ot, 0, reg);
            gen_op_mov_TN_reg(ot, 1, rm);
            gen_op_mov_reg_T0(ot, rm);
            gen_op_mov_reg_T1(ot, reg);
        } else {
            gen_lea_modrm(env, s, modrm, &reg_addr, &offset_addr);
            gen_op_mov_TN_reg(ot, 0, reg);
            /* for xchg, lock is implicit */
            if (!(prefixes & PREFIX_LOCK))
                gen_helper_lock();
            gen_op_ld_T1_A0(ot + s->mem_index);
            gen_op_st_T0_A0(ot + s->mem_index);
            if (!(prefixes & PREFIX_LOCK))
                gen_helper_unlock();
            gen_op_mov_reg_T1(ot, reg);
        break;
    case 0xc4: /* les Gv */
        /* In CODE64 this is VEX3; see above.  */
        op = R_ES;
        goto do_lxx;
    case 0xc5: /* lds Gv */
        /* In CODE64 this is VEX2; see above.  */
        op = R_DS;
        goto do_lxx;
    case 0x1b2: /* lss Gv */
        op = R_SS;
        goto do_lxx;
    case 0x1b4: /* lfs Gv */
        op = R_FS;
        goto do_lxx;
    case 0x1b5: /* lgs Gv */
        op = R_GS;
    do_lxx:
        ot = dflag ? OT_LONG : OT_WORD;
        modrm = cpu_ldub_code(env, s->pc++);
        reg = ((modrm >> 3) & 7) | rex_r;
        mod = (modrm >> 6) & 3;
        if (mod == 3)
        gen_lea_modrm(env, s, modrm, &reg_addr, &offset_addr);
        gen_op_ld_T1_A0(ot + s->mem_index);
        gen_add_A0_im(s, 1 << (ot - OT_WORD + 1));
        /* load the segment first to handle exceptions properly */
        gen_op_ldu_T0_A0(OT_WORD + s->mem_index);
        gen_movl_seg_T0(s, op, pc_start - s->cs_base);
        /* then put the data */
        gen_op_mov_reg_T1(ot, reg);
        if (s->is_jmp) {
            gen_jmp_im(s->pc - s->cs_base);
            gen_eob(s);
        break;
        /************************/
        /* shifts */
    case 0xc0:
    case 0xc1:
        /* shift Ev,Ib */
        shift = 2;
    grp2:
        {
            if ((b & 1) == 0)
                ot = OT_BYTE;
            else
                ot = dflag + OT_WORD;
            modrm = cpu_ldub_code(env, s->pc++);
            mod = (modrm >> 6) & 3;
            op = (modrm >> 3) & 7;
            if (mod != 3) {
                if (shift == 2) {
                    s->rip_offset = 1;
                gen_lea_modrm(env, s, modrm, &reg_addr, &offset_addr);
                opreg = OR_TMP0;
            } else {
                opreg = (modrm & 7) | REX_B(s);
            /* simpler op */
            if (shift == 0) {
                gen_shift(s, op, ot, opreg, OR_ECX);
            } else {
                if (shift == 2) {
                    shift = cpu_ldub_code(env, s->pc++);
                gen_shifti(s, op, ot, opreg, shift);
        break;
    case 0xd0:
    case 0xd1:
        /* shift Ev,1 */
        shift = 1;
        goto grp2;
    case 0xd2:
    case 0xd3:
        /* shift Ev,cl */
        shift = 0;
        goto grp2;
    case 0x1a4: /* shld imm */
        op = 0;
        shift = 1;
        goto do_shiftd;
    case 0x1a5: /* shld cl */
        op = 0;
        shift = 0;
        goto do_shiftd;
    case 0x1ac: /* shrd imm */
        op = 1;
        shift = 1;
        goto do_shiftd;
    case 0x1ad: /* shrd cl */
        op = 1;
        shift = 0;
    do_shiftd:
        ot = dflag + OT_WORD;
        modrm = cpu_ldub_code(env, s->pc++);
        mod = (modrm >> 6) & 3;
        rm = (modrm & 7) | REX_B(s);
        reg = ((modrm >> 3) & 7) | rex_r;
        if (mod != 3) {
            gen_lea_modrm(env, s, modrm, &reg_addr, &offset_addr);
            opreg = OR_TMP0;
        } else {
            opreg = rm;
        gen_op_mov_TN_reg(ot, 1, reg);
        if (shift) {
            TCGv imm = tcg_const_tl(cpu_ldub_code(env, s->pc++));
            gen_shiftd_rm_T1(s, ot, opreg, op, imm);
            tcg_temp_free(imm);
        } else {
            gen_shiftd_rm_T1(s, ot, opreg, op, cpu_regs[R_ECX]);
        break;
        /************************/
        /* floats */
    case 0xd8 ... 0xdf:
        if (s->flags & (HF_EM_MASK | HF_TS_MASK)) {
            /* if CR0.EM or CR0.TS are set, generate an FPU exception */
            /* XXX: what to do if illegal op ? */
            gen_exception(s, EXCP07_PREX, pc_start - s->cs_base);
            break;
        modrm = cpu_ldub_code(env, s->pc++);
        mod = (modrm >> 6) & 3;
        rm = modrm & 7;
        op = ((b & 7) << 3) | ((modrm >> 3) & 7);
        if (mod != 3) {
            /* memory op */
            gen_lea_modrm(env, s, modrm, &reg_addr, &offset_addr);
            switch(op) {
            case 0x00 ... 0x07: /* fxxxs */
            case 0x10 ... 0x17: /* fixxxl */
            case 0x20 ... 0x27: /* fxxxl */
            case 0x30 ... 0x37: /* fixxx */
                {
                    int op1;
                    op1 = op & 7;
                    switch(op >> 4) {
                    case 0:
                        gen_op_ld_T0_A0(OT_LONG + s->mem_index);
                        tcg_gen_trunc_tl_i32(cpu_tmp2_i32, cpu_T[0]);
                        gen_helper_flds_FT0(cpu_env, cpu_tmp2_i32);
                        break;
                    case 1:
                        gen_op_ld_T0_A0(OT_LONG + s->mem_index);
                        tcg_gen_trunc_tl_i32(cpu_tmp2_i32, cpu_T[0]);
                        gen_helper_fildl_FT0(cpu_env, cpu_tmp2_i32);
                        break;
                    case 2:
                        tcg_gen_qemu_ld64(cpu_tmp1_i64, cpu_A0, 
                                          (s->mem_index >> 2) - 1);
                        gen_helper_fldl_FT0(cpu_env, cpu_tmp1_i64);
                        break;
                    case 3:
                    default:
                        gen_op_lds_T0_A0(OT_WORD + s->mem_index);
                        tcg_gen_trunc_tl_i32(cpu_tmp2_i32, cpu_T[0]);
                        gen_helper_fildl_FT0(cpu_env, cpu_tmp2_i32);
                        break;
                    gen_helper_fp_arith_ST0_FT0(op1);
                    if (op1 == 3) {
                        /* fcomp needs pop */
                        gen_helper_fpop(cpu_env);
                break;
            case 0x08: /* flds */
            case 0x0a: /* fsts */
            case 0x0b: /* fstps */
            case 0x18 ... 0x1b: /* fildl, fisttpl, fistl, fistpl */
            case 0x28 ... 0x2b: /* fldl, fisttpll, fstl, fstpl */
            case 0x38 ... 0x3b: /* filds, fisttps, fists, fistps */
                switch(op & 7) {
                case 0:
                    switch(op >> 4) {
                    case 0:
                        gen_op_ld_T0_A0(OT_LONG + s->mem_index);
                        tcg_gen_trunc_tl_i32(cpu_tmp2_i32, cpu_T[0]);
                        gen_helper_flds_ST0(cpu_env, cpu_tmp2_i32);
                        break;
                    case 1:
                        gen_op_ld_T0_A0(OT_LONG + s->mem_index);
                        tcg_gen_trunc_tl_i32(cpu_tmp2_i32, cpu_T[0]);
                        gen_helper_fildl_ST0(cpu_env, cpu_tmp2_i32);
                        break;
                    case 2:
                        tcg_gen_qemu_ld64(cpu_tmp1_i64, cpu_A0, 
                                          (s->mem_index >> 2) - 1);
                        gen_helper_fldl_ST0(cpu_env, cpu_tmp1_i64);
                        break;
                    case 3:
                    default:
                        gen_op_lds_T0_A0(OT_WORD + s->mem_index);
                        tcg_gen_trunc_tl_i32(cpu_tmp2_i32, cpu_T[0]);
                        gen_helper_fildl_ST0(cpu_env, cpu_tmp2_i32);
                        break;
                    break;
                case 1:
                    /* XXX: the corresponding CPUID bit must be tested ! */
                    switch(op >> 4) {
                    case 1:
                        gen_helper_fisttl_ST0(cpu_tmp2_i32, cpu_env);
                        tcg_gen_extu_i32_tl(cpu_T[0], cpu_tmp2_i32);
                        gen_op_st_T0_A0(OT_LONG + s->mem_index);
                        break;
                    case 2:
                        gen_helper_fisttll_ST0(cpu_tmp1_i64, cpu_env);
                        tcg_gen_qemu_st64(cpu_tmp1_i64, cpu_A0, 
                                          (s->mem_index >> 2) - 1);
                        break;
                    case 3:
                    default:
                        gen_helper_fistt_ST0(cpu_tmp2_i32, cpu_env);
                        tcg_gen_extu_i32_tl(cpu_T[0], cpu_tmp2_i32);
                        gen_op_st_T0_A0(OT_WORD + s->mem_index);
                        break;
                    gen_helper_fpop(cpu_env);
                    break;
                default:
                    switch(op >> 4) {
                    case 0:
                        gen_helper_fsts_ST0(cpu_tmp2_i32, cpu_env);
                        tcg_gen_extu_i32_tl(cpu_T[0], cpu_tmp2_i32);
                        gen_op_st_T0_A0(OT_LONG + s->mem_index);
                        break;
                    case 1:
                        gen_helper_fistl_ST0(cpu_tmp2_i32, cpu_env);
                        tcg_gen_extu_i32_tl(cpu_T[0], cpu_tmp2_i32);
                        gen_op_st_T0_A0(OT_LONG + s->mem_index);
                        break;
                    case 2:
                        gen_helper_fstl_ST0(cpu_tmp1_i64, cpu_env);
                        tcg_gen_qemu_st64(cpu_tmp1_i64, cpu_A0, 
                                          (s->mem_index >> 2) - 1);
                        break;
                    case 3:
                    default:
                        gen_helper_fist_ST0(cpu_tmp2_i32, cpu_env);
                        tcg_gen_extu_i32_tl(cpu_T[0], cpu_tmp2_i32);
                        gen_op_st_T0_A0(OT_WORD + s->mem_index);
                        break;
                    if ((op & 7) == 3)
                        gen_helper_fpop(cpu_env);
                    break;
                break;
            case 0x0c: /* fldenv mem */
                gen_update_cc_op(s);
                gen_jmp_im(pc_start - s->cs_base);
                gen_helper_fldenv(cpu_env, cpu_A0, tcg_const_i32(s->dflag));
                break;
            case 0x0d: /* fldcw mem */
                gen_op_ld_T0_A0(OT_WORD + s->mem_index);
                tcg_gen_trunc_tl_i32(cpu_tmp2_i32, cpu_T[0]);
                gen_helper_fldcw(cpu_env, cpu_tmp2_i32);
                break;
            case 0x0e: /* fnstenv mem */
                gen_update_cc_op(s);
                gen_jmp_im(pc_start - s->cs_base);
                gen_helper_fstenv(cpu_env, cpu_A0, tcg_const_i32(s->dflag));
                break;
            case 0x0f: /* fnstcw mem */
                gen_helper_fnstcw(cpu_tmp2_i32, cpu_env);
                tcg_gen_extu_i32_tl(cpu_T[0], cpu_tmp2_i32);
                gen_op_st_T0_A0(OT_WORD + s->mem_index);
                break;
            case 0x1d: /* fldt mem */
                gen_update_cc_op(s);
                gen_jmp_im(pc_start - s->cs_base);
                gen_helper_fldt_ST0(cpu_env, cpu_A0);
                break;
            case 0x1f: /* fstpt mem */
                gen_update_cc_op(s);
                gen_jmp_im(pc_start - s->cs_base);
                gen_helper_fstt_ST0(cpu_env, cpu_A0);
                gen_helper_fpop(cpu_env);
                break;
            case 0x2c: /* frstor mem */
                gen_update_cc_op(s);
                gen_jmp_im(pc_start - s->cs_base);
                gen_helper_frstor(cpu_env, cpu_A0, tcg_const_i32(s->dflag));
                break;
            case 0x2e: /* fnsave mem */
                gen_update_cc_op(s);
                gen_jmp_im(pc_start - s->cs_base);
                gen_helper_fsave(cpu_env, cpu_A0, tcg_const_i32(s->dflag));
                break;
            case 0x2f: /* fnstsw mem */
                gen_helper_fnstsw(cpu_tmp2_i32, cpu_env);
                tcg_gen_extu_i32_tl(cpu_T[0], cpu_tmp2_i32);
                gen_op_st_T0_A0(OT_WORD + s->mem_index);
                break;
            case 0x3c: /* fbld */
                gen_update_cc_op(s);
                gen_jmp_im(pc_start - s->cs_base);
                gen_helper_fbld_ST0(cpu_env, cpu_A0);
                break;
            case 0x3e: /* fbstp */
                gen_update_cc_op(s);
                gen_jmp_im(pc_start - s->cs_base);
                gen_helper_fbst_ST0(cpu_env, cpu_A0);
                gen_helper_fpop(cpu_env);
                break;
            case 0x3d: /* fildll */
                tcg_gen_qemu_ld64(cpu_tmp1_i64, cpu_A0, 
                                  (s->mem_index >> 2) - 1);
                gen_helper_fildll_ST0(cpu_env, cpu_tmp1_i64);
                break;
            case 0x3f: /* fistpll */
                gen_helper_fistll_ST0(cpu_tmp1_i64, cpu_env);
                tcg_gen_qemu_st64(cpu_tmp1_i64, cpu_A0, 
                                  (s->mem_index >> 2) - 1);
                gen_helper_fpop(cpu_env);
                break;
            default:
        } else {
            /* register float ops */
            opreg = rm;
            switch(op) {
            case 0x08: /* fld sti */
                gen_helper_fpush(cpu_env);
                gen_helper_fmov_ST0_STN(cpu_env,
                                        tcg_const_i32((opreg + 1) & 7));
                break;
            case 0x09: /* fxchg sti */
            case 0x29: /* fxchg4 sti, undocumented op */
            case 0x39: /* fxchg7 sti, undocumented op */
                gen_helper_fxchg_ST0_STN(cpu_env, tcg_const_i32(opreg));
                break;
            case 0x0a: /* grp d9/2 */
                switch(rm) {
                case 0: /* fnop */
                    /* check exceptions (FreeBSD FPU probe) */
                    gen_update_cc_op(s);
                    gen_jmp_im(pc_start - s->cs_base);
                    gen_helper_fwait(cpu_env);
                    break;
                default:
                break;
            case 0x0c: /* grp d9/4 */
                switch(rm) {
                case 0: /* fchs */
                    gen_helper_fchs_ST0(cpu_env);
                    break;
                case 1: /* fabs */
                    gen_helper_fabs_ST0(cpu_env);
                    break;
                case 4: /* ftst */
                    gen_helper_fldz_FT0(cpu_env);
                    gen_helper_fcom_ST0_FT0(cpu_env);
                    break;
                case 5: /* fxam */
                    gen_helper_fxam_ST0(cpu_env);
                    break;
                default:
                break;
            case 0x0d: /* grp d9/5 */
                {
                    switch(rm) {
                    case 0:
                        gen_helper_fpush(cpu_env);
                        gen_helper_fld1_ST0(cpu_env);
                        break;
                    case 1:
                        gen_helper_fpush(cpu_env);
                        gen_helper_fldl2t_ST0(cpu_env);
                        break;
                    case 2:
                        gen_helper_fpush(cpu_env);
                        gen_helper_fldl2e_ST0(cpu_env);
                        break;
                    case 3:
                        gen_helper_fpush(cpu_env);
                        gen_helper_fldpi_ST0(cpu_env);
                        break;
                    case 4:
                        gen_helper_fpush(cpu_env);
                        gen_helper_fldlg2_ST0(cpu_env);
                        break;
                    case 5:
                        gen_helper_fpush(cpu_env);
                        gen_helper_fldln2_ST0(cpu_env);
                        break;
                    case 6:
                        gen_helper_fpush(cpu_env);
                        gen_helper_fldz_ST0(cpu_env);
                        break;
                    default:
                break;
            case 0x0e: /* grp d9/6 */
                switch(rm) {
                case 0: /* f2xm1 */
                    gen_helper_f2xm1(cpu_env);
                    break;
                case 1: /* fyl2x */
                    gen_helper_fyl2x(cpu_env);
                    break;
                case 2: /* fptan */
                    gen_helper_fptan(cpu_env);
                    break;
                case 3: /* fpatan */
                    gen_helper_fpatan(cpu_env);
                    break;
                case 4: /* fxtract */
                    gen_helper_fxtract(cpu_env);
                    break;
                case 5: /* fprem1 */
                    gen_helper_fprem1(cpu_env);
                    break;
                case 6: /* fdecstp */
                    gen_helper_fdecstp(cpu_env);
                    break;
                default:
                case 7: /* fincstp */
                    gen_helper_fincstp(cpu_env);
                    break;
                break;
            case 0x0f: /* grp d9/7 */
                switch(rm) {
                case 0: /* fprem */
                    gen_helper_fprem(cpu_env);
                    break;
                case 1: /* fyl2xp1 */
                    gen_helper_fyl2xp1(cpu_env);
                    break;
                case 2: /* fsqrt */
                    gen_helper_fsqrt(cpu_env);
                    break;
                case 3: /* fsincos */
                    gen_helper_fsincos(cpu_env);
                    break;
                case 5: /* fscale */
                    gen_helper_fscale(cpu_env);
                    break;
                case 4: /* frndint */
                    gen_helper_frndint(cpu_env);
                    break;
                case 6: /* fsin */
                    gen_helper_fsin(cpu_env);
                    break;
                default:
                case 7: /* fcos */
                    gen_helper_fcos(cpu_env);
                    break;
                break;
            case 0x00: case 0x01: case 0x04 ... 0x07: /* fxxx st, sti */
            case 0x20: case 0x21: case 0x24 ... 0x27: /* fxxx sti, st */
            case 0x30: case 0x31: case 0x34 ... 0x37: /* fxxxp sti, st */
                {
                    int op1;
                    op1 = op & 7;
                    if (op >= 0x20) {
                        gen_helper_fp_arith_STN_ST0(op1, opreg);
                        if (op >= 0x30)
                            gen_helper_fpop(cpu_env);
                    } else {
                        gen_helper_fmov_FT0_STN(cpu_env, tcg_const_i32(opreg));
                        gen_helper_fp_arith_ST0_FT0(op1);
                break;
            case 0x02: /* fcom */
            case 0x22: /* fcom2, undocumented op */
                gen_helper_fmov_FT0_STN(cpu_env, tcg_const_i32(opreg));
                gen_helper_fcom_ST0_FT0(cpu_env);
                break;
            case 0x03: /* fcomp */
            case 0x23: /* fcomp3, undocumented op */
            case 0x32: /* fcomp5, undocumented op */
                gen_helper_fmov_FT0_STN(cpu_env, tcg_const_i32(opreg));
                gen_helper_fcom_ST0_FT0(cpu_env);
                gen_helper_fpop(cpu_env);
                break;
            case 0x15: /* da/5 */
                switch(rm) {
                case 1: /* fucompp */
                    gen_helper_fmov_FT0_STN(cpu_env, tcg_const_i32(1));
                    gen_helper_fucom_ST0_FT0(cpu_env);
                    gen_helper_fpop(cpu_env);
                    gen_helper_fpop(cpu_env);
                    break;
                default:
                break;
            case 0x1c:
                switch(rm) {
                case 0: /* feni (287 only, just do nop here) */
                    break;
                case 1: /* fdisi (287 only, just do nop here) */
                    break;
                case 2: /* fclex */
                    gen_helper_fclex(cpu_env);
                    break;
                case 3: /* fninit */
                    gen_helper_fninit(cpu_env);
                    break;
                case 4: /* fsetpm (287 only, just do nop here) */
                    break;
                default:
                break;
            case 0x1d: /* fucomi */
                gen_update_cc_op(s);
                gen_helper_fmov_FT0_STN(cpu_env, tcg_const_i32(opreg));
                gen_helper_fucomi_ST0_FT0(cpu_env);
                set_cc_op(s, CC_OP_EFLAGS);
                break;
            case 0x1e: /* fcomi */
                gen_update_cc_op(s);
                gen_helper_fmov_FT0_STN(cpu_env, tcg_const_i32(opreg));
                gen_helper_fcomi_ST0_FT0(cpu_env);
                set_cc_op(s, CC_OP_EFLAGS);
                break;
            case 0x28: /* ffree sti */
                gen_helper_ffree_STN(cpu_env, tcg_const_i32(opreg));
                break;
            case 0x2a: /* fst sti */
                gen_helper_fmov_STN_ST0(cpu_env, tcg_const_i32(opreg));
                break;
            case 0x2b: /* fstp sti */
            case 0x0b: /* fstp1 sti, undocumented op */
            case 0x3a: /* fstp8 sti, undocumented op */
            case 0x3b: /* fstp9 sti, undocumented op */
                gen_helper_fmov_STN_ST0(cpu_env, tcg_const_i32(opreg));
                gen_helper_fpop(cpu_env);
                break;
            case 0x2c: /* fucom st(i) */
                gen_helper_fmov_FT0_STN(cpu_env, tcg_const_i32(opreg));
                gen_helper_fucom_ST0_FT0(cpu_env);
                break;
            case 0x2d: /* fucomp st(i) */
                gen_helper_fmov_FT0_STN(cpu_env, tcg_const_i32(opreg));
                gen_helper_fucom_ST0_FT0(cpu_env);
                gen_helper_fpop(cpu_env);
                break;
            case 0x33: /* de/3 */
                switch(rm) {
                case 1: /* fcompp */
                    gen_helper_fmov_FT0_STN(cpu_env, tcg_const_i32(1));
                    gen_helper_fcom_ST0_FT0(cpu_env);
                    gen_helper_fpop(cpu_env);
                    gen_helper_fpop(cpu_env);
                    break;
                default:
                break;
            case 0x38: /* ffreep sti, undocumented op */
                gen_helper_ffree_STN(cpu_env, tcg_const_i32(opreg));
                gen_helper_fpop(cpu_env);
                break;
            case 0x3c: /* df/4 */
                switch(rm) {
                case 0:
                    gen_helper_fnstsw(cpu_tmp2_i32, cpu_env);
                    tcg_gen_extu_i32_tl(cpu_T[0], cpu_tmp2_i32);
                    gen_op_mov_reg_T0(OT_WORD, R_EAX);
                    break;
                default:
                break;
            case 0x3d: /* fucomip */
                gen_update_cc_op(s);
                gen_helper_fmov_FT0_STN(cpu_env, tcg_const_i32(opreg));
                gen_helper_fucomi_ST0_FT0(cpu_env);
                gen_helper_fpop(cpu_env);
                set_cc_op(s, CC_OP_EFLAGS);
                break;
            case 0x3e: /* fcomip */
                gen_update_cc_op(s);
                gen_helper_fmov_FT0_STN(cpu_env, tcg_const_i32(opreg));
                gen_helper_fcomi_ST0_FT0(cpu_env);
                gen_helper_fpop(cpu_env);
                set_cc_op(s, CC_OP_EFLAGS);
                break;
            case 0x10 ... 0x13: /* fcmovxx */
            case 0x18 ... 0x1b:
                {
                    int op1, l1;
                    static const uint8_t fcmov_cc[8] = {
                        (JCC_B << 1),
                        (JCC_Z << 1),
                        (JCC_BE << 1),
                        (JCC_P << 1),
                    };
                    op1 = fcmov_cc[op & 3] | (((op >> 3) & 1) ^ 1);
                    l1 = gen_new_label();
                    gen_jcc1_noeob(s, op1, l1);
                    gen_helper_fmov_ST0_STN(cpu_env, tcg_const_i32(opreg));
                    gen_set_label(l1);
                break;
            default:
        break;
        /************************/
        /* string ops */
    case 0xa4: /* movsS */
    case 0xa5:
        if ((b & 1) == 0)
            ot = OT_BYTE;
        else
            ot = dflag + OT_WORD;
        if (prefixes & (PREFIX_REPZ | PREFIX_REPNZ)) {
            gen_repz_movs(s, ot, pc_start - s->cs_base, s->pc - s->cs_base);
        } else {
            gen_movs(s, ot);
        break;
    case 0xaa: /* stosS */
    case 0xab:
        if ((b & 1) == 0)
            ot = OT_BYTE;
        else
            ot = dflag + OT_WORD;
        if (prefixes & (PREFIX_REPZ | PREFIX_REPNZ)) {
            gen_repz_stos(s, ot, pc_start - s->cs_base, s->pc - s->cs_base);
        } else {
            gen_stos(s, ot);
        break;
    case 0xac: /* lodsS */
    case 0xad:
        if ((b & 1) == 0)
            ot = OT_BYTE;
        else
            ot = dflag + OT_WORD;
        if (prefixes & (PREFIX_REPZ | PREFIX_REPNZ)) {
            gen_repz_lods(s, ot, pc_start - s->cs_base, s->pc - s->cs_base);
        } else {
            gen_lods(s, ot);
        break;
    case 0xae: /* scasS */
    case 0xaf:
        if ((b & 1) == 0)
            ot = OT_BYTE;
        else
            ot = dflag + OT_WORD;
        if (prefixes & PREFIX_REPNZ) {
            gen_repz_scas(s, ot, pc_start - s->cs_base, s->pc - s->cs_base, 1);
        } else if (prefixes & PREFIX_REPZ) {
            gen_repz_scas(s, ot, pc_start - s->cs_base, s->pc - s->cs_base, 0);
        } else {
            gen_scas(s, ot);
        break;
    case 0xa6: /* cmpsS */
    case 0xa7:
        if ((b & 1) == 0)
            ot = OT_BYTE;
        else
            ot = dflag + OT_WORD;
        if (prefixes & PREFIX_REPNZ) {
            gen_repz_cmps(s, ot, pc_start - s->cs_base, s->pc - s->cs_base, 1);
        } else if (prefixes & PREFIX_REPZ) {
            gen_repz_cmps(s, ot, pc_start - s->cs_base, s->pc - s->cs_base, 0);
        } else {
            gen_cmps(s, ot);
        break;
    case 0x6c: /* insS */
    case 0x6d:
        if ((b & 1) == 0)
            ot = OT_BYTE;
        else
            ot = dflag ? OT_LONG : OT_WORD;
        gen_op_mov_TN_reg(OT_WORD, 0, R_EDX);
        gen_op_andl_T0_ffff();
        gen_check_io(s, ot, pc_start - s->cs_base, 
                     SVM_IOIO_TYPE_MASK | svm_is_rep(prefixes) | 4);
        if (prefixes & (PREFIX_REPZ | PREFIX_REPNZ)) {
            gen_repz_ins(s, ot, pc_start - s->cs_base, s->pc - s->cs_base);
        } else {
            gen_ins(s, ot);
            if (use_icount) {
                gen_jmp(s, s->pc - s->cs_base);
        break;
    case 0x6e: /* outsS */
    case 0x6f:
        if ((b & 1) == 0)
            ot = OT_BYTE;
        else
            ot = dflag ? OT_LONG : OT_WORD;
        gen_op_mov_TN_reg(OT_WORD, 0, R_EDX);
        gen_op_andl_T0_ffff();
        gen_check_io(s, ot, pc_start - s->cs_base,
                     svm_is_rep(prefixes) | 4);
        if (prefixes & (PREFIX_REPZ | PREFIX_REPNZ)) {
            gen_repz_outs(s, ot, pc_start - s->cs_base, s->pc - s->cs_base);
        } else {
            gen_outs(s, ot);
            if (use_icount) {
                gen_jmp(s, s->pc - s->cs_base);
        break;
        /************************/
        /* port I/O */
    case 0xe4:
    case 0xe5:
        if ((b & 1) == 0)
            ot = OT_BYTE;
        else
            ot = dflag ? OT_LONG : OT_WORD;
        val = cpu_ldub_code(env, s->pc++);
        gen_op_movl_T0_im(val);
        gen_check_io(s, ot, pc_start - s->cs_base,
                     SVM_IOIO_TYPE_MASK | svm_is_rep(prefixes));
        if (use_icount)
            gen_io_start();
        tcg_gen_trunc_tl_i32(cpu_tmp2_i32, cpu_T[0]);
        gen_helper_in_func(ot, cpu_T[1], cpu_tmp2_i32);
        gen_op_mov_reg_T1(ot, R_EAX);
        if (use_icount) {
            gen_io_end();
            gen_jmp(s, s->pc - s->cs_base);
        break;
    case 0xe6:
    case 0xe7:
        if ((b & 1) == 0)
            ot = OT_BYTE;
        else
            ot = dflag ? OT_LONG : OT_WORD;
        val = cpu_ldub_code(env, s->pc++);
        gen_op_movl_T0_im(val);
        gen_check_io(s, ot, pc_start - s->cs_base,
                     svm_is_rep(prefixes));
        gen_op_mov_TN_reg(ot, 1, R_EAX);
        if (use_icount)
            gen_io_start();
        tcg_gen_trunc_tl_i32(cpu_tmp2_i32, cpu_T[0]);
        tcg_gen_trunc_tl_i32(cpu_tmp3_i32, cpu_T[1]);
        gen_helper_out_func(ot, cpu_tmp2_i32, cpu_tmp3_i32);
        if (use_icount) {
            gen_io_end();
            gen_jmp(s, s->pc - s->cs_base);
        break;
    case 0xec:
    case 0xed:
        if ((b & 1) == 0)
            ot = OT_BYTE;
        else
            ot = dflag ? OT_LONG : OT_WORD;
        gen_op_mov_TN_reg(OT_WORD, 0, R_EDX);
        gen_op_andl_T0_ffff();
        gen_check_io(s, ot, pc_start - s->cs_base,
                     SVM_IOIO_TYPE_MASK | svm_is_rep(prefixes));
        if (use_icount)
            gen_io_start();
        tcg_gen_trunc_tl_i32(cpu_tmp2_i32, cpu_T[0]);
        gen_helper_in_func(ot, cpu_T[1], cpu_tmp2_i32);
        gen_op_mov_reg_T1(ot, R_EAX);
        if (use_icount) {
            gen_io_end();
            gen_jmp(s, s->pc - s->cs_base);
        break;
    case 0xee:
    case 0xef:
        if ((b & 1) == 0)
            ot = OT_BYTE;
        else
            ot = dflag ? OT_LONG : OT_WORD;
        gen_op_mov_TN_reg(OT_WORD, 0, R_EDX);
        gen_op_andl_T0_ffff();
        gen_check_io(s, ot, pc_start - s->cs_base,
                     svm_is_rep(prefixes));
        gen_op_mov_TN_reg(ot, 1, R_EAX);
        if (use_icount)
            gen_io_start();
        tcg_gen_trunc_tl_i32(cpu_tmp2_i32, cpu_T[0]);
        tcg_gen_trunc_tl_i32(cpu_tmp3_i32, cpu_T[1]);
        gen_helper_out_func(ot, cpu_tmp2_i32, cpu_tmp3_i32);
        if (use_icount) {
            gen_io_end();
            gen_jmp(s, s->pc - s->cs_base);
        break;
        /************************/
        /* control */
    case 0xc2: /* ret im */
        val = cpu_ldsw_code(env, s->pc);
        s->pc += 2;
        gen_pop_T0(s);
        if (CODE64(s) && s->dflag)
            s->dflag = 2;
        gen_stack_update(s, val + (2 << s->dflag));
        if (s->dflag == 0)
            gen_op_andl_T0_ffff();
        gen_op_jmp_T0();
        gen_eob(s);
        break;
    case 0xc3: /* ret */
        gen_pop_T0(s);
        gen_pop_update(s);
        if (s->dflag == 0)
            gen_op_andl_T0_ffff();
        gen_op_jmp_T0();
        gen_eob(s);
        break;
    case 0xca: /* lret im */
        val = cpu_ldsw_code(env, s->pc);
        s->pc += 2;
    do_lret:
        if (s->pe && !s->vm86) {
            gen_update_cc_op(s);
            gen_jmp_im(pc_start - s->cs_base);
            gen_helper_lret_protected(cpu_env, tcg_const_i32(s->dflag),
                                      tcg_const_i32(val));
        } else {
            gen_stack_A0(s);
            /* pop offset */
            gen_op_ld_T0_A0(1 + s->dflag + s->mem_index);
            if (s->dflag == 0)
                gen_op_andl_T0_ffff();
            /* NOTE: keeping EIP updated is not a problem in case of
               exception */
            gen_op_jmp_T0();
            /* pop selector */
            gen_op_addl_A0_im(2 << s->dflag);
            gen_op_ld_T0_A0(1 + s->dflag + s->mem_index);
            gen_op_movl_seg_T0_vm(R_CS);
            /* add stack offset */
            gen_stack_update(s, val + (4 << s->dflag));
        gen_eob(s);
        break;
    case 0xcb: /* lret */
        val = 0;
        goto do_lret;
    case 0xcf: /* iret */
        gen_svm_check_intercept(s, pc_start, SVM_EXIT_IRET);
        if (!s->pe) {
            /* real mode */
            gen_helper_iret_real(cpu_env, tcg_const_i32(s->dflag));
            set_cc_op(s, CC_OP_EFLAGS);
        } else if (s->vm86) {
            if (s->iopl != 3) {
                gen_exception(s, EXCP0D_GPF, pc_start - s->cs_base);
            } else {
                gen_helper_iret_real(cpu_env, tcg_const_i32(s->dflag));
                set_cc_op(s, CC_OP_EFLAGS);
        } else {
            gen_update_cc_op(s);
            gen_jmp_im(pc_start - s->cs_base);
            gen_helper_iret_protected(cpu_env, tcg_const_i32(s->dflag),
                                      tcg_const_i32(s->pc - s->cs_base));
            set_cc_op(s, CC_OP_EFLAGS);
        gen_eob(s);
        break;
    case 0xe8: /* call im */
        {
            if (dflag)
                tval = (int32_t)insn_get(env, s, OT_LONG);
            else
                tval = (int16_t)insn_get(env, s, OT_WORD);
            next_eip = s->pc - s->cs_base;
            tval += next_eip;
            if (s->dflag == 0)
                tval &= 0xffff;
            else if(!CODE64(s))
                tval &= 0xffffffff;
            gen_movtl_T0_im(next_eip);
            gen_push_T0(s);
            gen_jmp(s, tval);
        break;
    case 0x9a: /* lcall im */
        {
            unsigned int selector, offset;
            if (CODE64(s))
            ot = dflag ? OT_LONG : OT_WORD;
            offset = insn_get(env, s, ot);
            selector = insn_get(env, s, OT_WORD);
            gen_op_movl_T0_im(selector);
            gen_op_movl_T1_imu(offset);
        goto do_lcall;
    case 0xe9: /* jmp im */
        if (dflag)
            tval = (int32_t)insn_get(env, s, OT_LONG);
        else
            tval = (int16_t)insn_get(env, s, OT_WORD);
        tval += s->pc - s->cs_base;
        if (s->dflag == 0)
            tval &= 0xffff;
        else if(!CODE64(s))
            tval &= 0xffffffff;
        gen_jmp(s, tval);
        break;
    case 0xea: /* ljmp im */
        {
            unsigned int selector, offset;
            if (CODE64(s))
            ot = dflag ? OT_LONG : OT_WORD;
            offset = insn_get(env, s, ot);
            selector = insn_get(env, s, OT_WORD);
            gen_op_movl_T0_im(selector);
            gen_op_movl_T1_imu(offset);
        goto do_ljmp;
    case 0xeb: /* jmp Jb */
        tval = (int8_t)insn_get(env, s, OT_BYTE);
        tval += s->pc - s->cs_base;
        if (s->dflag == 0)
            tval &= 0xffff;
        gen_jmp(s, tval);
        break;
    case 0x70 ... 0x7f: /* jcc Jb */
        tval = (int8_t)insn_get(env, s, OT_BYTE);
        goto do_jcc;
    case 0x180 ... 0x18f: /* jcc Jv */
        if (dflag) {
            tval = (int32_t)insn_get(env, s, OT_LONG);
        } else {
            tval = (int16_t)insn_get(env, s, OT_WORD);
    do_jcc:
        next_eip = s->pc - s->cs_base;
        tval += next_eip;
        if (s->dflag == 0)
            tval &= 0xffff;
        gen_jcc(s, b, tval, next_eip);
        break;
    case 0x190 ... 0x19f: /* setcc Gv */
        modrm = cpu_ldub_code(env, s->pc++);
        gen_setcc1(s, b, cpu_T[0]);
        gen_ldst_modrm(env, s, modrm, OT_BYTE, OR_TMP0, 1);
        break;
    case 0x140 ... 0x14f: /* cmov Gv, Ev */
        ot = dflag + OT_WORD;
        modrm = cpu_ldub_code(env, s->pc++);
        reg = ((modrm >> 3) & 7) | rex_r;
        gen_cmovcc1(env, s, ot, b, modrm, reg);
        break;
        /************************/
        /* flags */
    case 0x9c: /* pushf */
        gen_svm_check_intercept(s, pc_start, SVM_EXIT_PUSHF);
        if (s->vm86 && s->iopl != 3) {
            gen_exception(s, EXCP0D_GPF, pc_start - s->cs_base);
        } else {
            gen_update_cc_op(s);
            gen_helper_read_eflags(cpu_T[0], cpu_env);
            gen_push_T0(s);
        break;
    case 0x9d: /* popf */
        gen_svm_check_intercept(s, pc_start, SVM_EXIT_POPF);
        if (s->vm86 && s->iopl != 3) {
            gen_exception(s, EXCP0D_GPF, pc_start - s->cs_base);
        } else {
            gen_pop_T0(s);
            if (s->cpl == 0) {
                if (s->dflag) {
                    gen_helper_write_eflags(cpu_env, cpu_T[0],
                                            tcg_const_i32((TF_MASK | AC_MASK |
                                                           ID_MASK | NT_MASK |
                                                           IF_MASK |
                                                           IOPL_MASK)));
                } else {
                    gen_helper_write_eflags(cpu_env, cpu_T[0],
                                            tcg_const_i32((TF_MASK | AC_MASK |
                                                           ID_MASK | NT_MASK |
                                                           IF_MASK | IOPL_MASK)
                                                          & 0xffff));
            } else {
                if (s->cpl <= s->iopl) {
                    if (s->dflag) {
                        gen_helper_write_eflags(cpu_env, cpu_T[0],
                                                tcg_const_i32((TF_MASK |
                                                               AC_MASK |
                                                               ID_MASK |
                                                               NT_MASK |
                                                               IF_MASK)));
                    } else {
                        gen_helper_write_eflags(cpu_env, cpu_T[0],
                                                tcg_const_i32((TF_MASK |
                                                               AC_MASK |
                                                               ID_MASK |
                                                               NT_MASK |
                                                               IF_MASK)
                                                              & 0xffff));
                } else {
                    if (s->dflag) {
                        gen_helper_write_eflags(cpu_env, cpu_T[0],
                                           tcg_const_i32((TF_MASK | AC_MASK |
                                                          ID_MASK | NT_MASK)));
                    } else {
                        gen_helper_write_eflags(cpu_env, cpu_T[0],
                                           tcg_const_i32((TF_MASK | AC_MASK |
                                                          ID_MASK | NT_MASK)
                                                         & 0xffff));
            gen_pop_update(s);
            set_cc_op(s, CC_OP_EFLAGS);
            /* abort translation because TF/AC flag may change */
            gen_jmp_im(s->pc - s->cs_base);
            gen_eob(s);
        break;
    case 0x9e: /* sahf */
        if (CODE64(s) && !(s->cpuid_ext3_features & CPUID_EXT3_LAHF_LM))
        gen_op_mov_TN_reg(OT_BYTE, 0, R_AH);
        gen_compute_eflags(s);
        tcg_gen_andi_tl(cpu_cc_src, cpu_cc_src, CC_O);
        tcg_gen_andi_tl(cpu_T[0], cpu_T[0], CC_S | CC_Z | CC_A | CC_P | CC_C);
        tcg_gen_or_tl(cpu_cc_src, cpu_cc_src, cpu_T[0]);
        break;
    case 0x9f: /* lahf */
        if (CODE64(s) && !(s->cpuid_ext3_features & CPUID_EXT3_LAHF_LM))
        gen_compute_eflags(s);
        /* Note: gen_compute_eflags() only gives the condition codes */
        tcg_gen_ori_tl(cpu_T[0], cpu_cc_src, 0x02);
        gen_op_mov_reg_T0(OT_BYTE, R_AH);
        break;
    case 0xf5: /* cmc */
        gen_compute_eflags(s);
        tcg_gen_xori_tl(cpu_cc_src, cpu_cc_src, CC_C);
        break;
    case 0xf8: /* clc */
        gen_compute_eflags(s);
        tcg_gen_andi_tl(cpu_cc_src, cpu_cc_src, ~CC_C);
        break;
    case 0xf9: /* stc */
        gen_compute_eflags(s);
        tcg_gen_ori_tl(cpu_cc_src, cpu_cc_src, CC_C);
        break;
    case 0xfc: /* cld */
        tcg_gen_movi_i32(cpu_tmp2_i32, 1);
        tcg_gen_st_i32(cpu_tmp2_i32, cpu_env, offsetof(CPUX86State, df));
        break;
    case 0xfd: /* std */
        tcg_gen_movi_i32(cpu_tmp2_i32, -1);
        tcg_gen_st_i32(cpu_tmp2_i32, cpu_env, offsetof(CPUX86State, df));
        break;
        /************************/
        /* bit operations */
    case 0x1ba: /* bt/bts/btr/btc Gv, im */
        ot = dflag + OT_WORD;
        modrm = cpu_ldub_code(env, s->pc++);
        op = (modrm >> 3) & 7;
        mod = (modrm >> 6) & 3;
        rm = (modrm & 7) | REX_B(s);
        if (mod != 3) {
            s->rip_offset = 1;
            gen_lea_modrm(env, s, modrm, &reg_addr, &offset_addr);
            gen_op_ld_T0_A0(ot + s->mem_index);
        } else {
            gen_op_mov_TN_reg(ot, 0, rm);
        /* load shift */
        val = cpu_ldub_code(env, s->pc++);
        gen_op_movl_T1_im(val);
        if (op < 4)
        op -= 4;
        goto bt_op;
    case 0x1a3: /* bt Gv, Ev */
        op = 0;
        goto do_btx;
    case 0x1ab: /* bts */
        op = 1;
        goto do_btx;
    case 0x1b3: /* btr */
        op = 2;
        goto do_btx;
    case 0x1bb: /* btc */
        op = 3;
    do_btx:
        ot = dflag + OT_WORD;
        modrm = cpu_ldub_code(env, s->pc++);
        reg = ((modrm >> 3) & 7) | rex_r;
        mod = (modrm >> 6) & 3;
        rm = (modrm & 7) | REX_B(s);
        gen_op_mov_TN_reg(OT_LONG, 1, reg);
        if (mod != 3) {
            gen_lea_modrm(env, s, modrm, &reg_addr, &offset_addr);
            /* specific case: we need to add a displacement */
            gen_exts(ot, cpu_T[1]);
            tcg_gen_sari_tl(cpu_tmp0, cpu_T[1], 3 + ot);
            tcg_gen_shli_tl(cpu_tmp0, cpu_tmp0, ot);
            tcg_gen_add_tl(cpu_A0, cpu_A0, cpu_tmp0);
            gen_op_ld_T0_A0(ot + s->mem_index);
        } else {
            gen_op_mov_TN_reg(ot, 0, rm);
    bt_op:
        tcg_gen_andi_tl(cpu_T[1], cpu_T[1], (1 << (3 + ot)) - 1);
        switch(op) {
        case 0:
            tcg_gen_shr_tl(cpu_cc_src, cpu_T[0], cpu_T[1]);
            tcg_gen_movi_tl(cpu_cc_dst, 0);
            break;
        case 1:
            tcg_gen_shr_tl(cpu_tmp4, cpu_T[0], cpu_T[1]);
            tcg_gen_movi_tl(cpu_tmp0, 1);
            tcg_gen_shl_tl(cpu_tmp0, cpu_tmp0, cpu_T[1]);
            tcg_gen_or_tl(cpu_T[0], cpu_T[0], cpu_tmp0);
            break;
        case 2:
            tcg_gen_shr_tl(cpu_tmp4, cpu_T[0], cpu_T[1]);
            tcg_gen_movi_tl(cpu_tmp0, 1);
            tcg_gen_shl_tl(cpu_tmp0, cpu_tmp0, cpu_T[1]);
            tcg_gen_not_tl(cpu_tmp0, cpu_tmp0);
            tcg_gen_and_tl(cpu_T[0], cpu_T[0], cpu_tmp0);
            break;
        default:
        case 3:
            tcg_gen_shr_tl(cpu_tmp4, cpu_T[0], cpu_T[1]);
            tcg_gen_movi_tl(cpu_tmp0, 1);
            tcg_gen_shl_tl(cpu_tmp0, cpu_tmp0, cpu_T[1]);
            tcg_gen_xor_tl(cpu_T[0], cpu_T[0], cpu_tmp0);
            break;
        set_cc_op(s, CC_OP_SARB + ot);
        if (op != 0) {
            if (mod != 3)
                gen_op_st_T0_A0(ot + s->mem_index);
            else
                gen_op_mov_reg_T0(ot, rm);
            tcg_gen_mov_tl(cpu_cc_src, cpu_tmp4);
            tcg_gen_movi_tl(cpu_cc_dst, 0);
        break;
    case 0x1bc: /* bsf / tzcnt */
    case 0x1bd: /* bsr / lzcnt */
        ot = dflag + OT_WORD;
        modrm = cpu_ldub_code(env, s->pc++);
        reg = ((modrm >> 3) & 7) | rex_r;
        gen_ldst_modrm(env, s, modrm, ot, OR_TMP0, 0);
        gen_extu(ot, cpu_T[0]);
        /* Note that lzcnt and tzcnt are in different extensions.  */
        if ((prefixes & PREFIX_REPZ)
            && (b & 1
                ? s->cpuid_ext3_features & CPUID_EXT3_ABM
                : s->cpuid_7_0_ebx_features & CPUID_7_0_EBX_BMI1)) {
            int size = 8 << ot;
            tcg_gen_mov_tl(cpu_cc_src, cpu_T[0]);
            if (b & 1) {
                /* For lzcnt, reduce the target_ulong result by the
                   number of zeros that we expect to find at the top.  */
                gen_helper_clz(cpu_T[0], cpu_T[0]);
                tcg_gen_subi_tl(cpu_T[0], cpu_T[0], TARGET_LONG_BITS - size);
            } else {
                /* For tzcnt, a zero input must return the operand size:
                   force all bits outside the operand size to 1.  */
                target_ulong mask = (target_ulong)-2 << (size - 1);
                tcg_gen_ori_tl(cpu_T[0], cpu_T[0], mask);
                gen_helper_ctz(cpu_T[0], cpu_T[0]);
            /* For lzcnt/tzcnt, C and Z bits are defined and are
               related to the result.  */
            gen_op_update1_cc();
            set_cc_op(s, CC_OP_BMILGB + ot);
        } else {
            /* For bsr/bsf, only the Z bit is defined and it is related
               to the input and not the result.  */
            tcg_gen_mov_tl(cpu_cc_dst, cpu_T[0]);
            set_cc_op(s, CC_OP_LOGICB + ot);
            if (b & 1) {
                /* For bsr, return the bit index of the first 1 bit,
                   not the count of leading zeros.  */
                gen_helper_clz(cpu_T[0], cpu_T[0]);
                tcg_gen_xori_tl(cpu_T[0], cpu_T[0], TARGET_LONG_BITS - 1);
            } else {
                gen_helper_ctz(cpu_T[0], cpu_T[0]);
            /* ??? The manual says that the output is undefined when the
               input is zero, but real hardware leaves it unchanged, and
               real programs appear to depend on that.  */
            tcg_gen_movi_tl(cpu_tmp0, 0);
            tcg_gen_movcond_tl(TCG_COND_EQ, cpu_T[0], cpu_cc_dst, cpu_tmp0,
                               cpu_regs[reg], cpu_T[0]);
        gen_op_mov_reg_T0(ot, reg);
        break;
        /************************/
        /* bcd */
    case 0x27: /* daa */
        if (CODE64(s))
        gen_update_cc_op(s);
        gen_helper_daa(cpu_env);
        set_cc_op(s, CC_OP_EFLAGS);
        break;
    case 0x2f: /* das */
        if (CODE64(s))
        gen_update_cc_op(s);
        gen_helper_das(cpu_env);
        set_cc_op(s, CC_OP_EFLAGS);
        break;
    case 0x37: /* aaa */
        if (CODE64(s))
        gen_update_cc_op(s);
        gen_helper_aaa(cpu_env);
        set_cc_op(s, CC_OP_EFLAGS);
        break;
    case 0x3f: /* aas */
        if (CODE64(s))
        gen_update_cc_op(s);
        gen_helper_aas(cpu_env);
        set_cc_op(s, CC_OP_EFLAGS);
        break;
    case 0xd4: /* aam */
        if (CODE64(s))
        val = cpu_ldub_code(env, s->pc++);
        if (val == 0) {
            gen_exception(s, EXCP00_DIVZ, pc_start - s->cs_base);
        } else {
            gen_helper_aam(cpu_env, tcg_const_i32(val));
            set_cc_op(s, CC_OP_LOGICB);
        break;
    case 0xd5: /* aad */
        if (CODE64(s))
        val = cpu_ldub_code(env, s->pc++);
        gen_helper_aad(cpu_env, tcg_const_i32(val));
        set_cc_op(s, CC_OP_LOGICB);
        break;
        /************************/
        /* misc */
    case 0x90: /* nop */
        /* XXX: correct lock test for all insn */
        if (prefixes & PREFIX_LOCK) {
        /* If REX_B is set, then this is xchg eax, r8d, not a nop.  */
        if (REX_B(s)) {
            goto do_xchg_reg_eax;
        if (prefixes & PREFIX_REPZ) {
            gen_svm_check_intercept(s, pc_start, SVM_EXIT_PAUSE);
        break;
    case 0x9b: /* fwait */
        if ((s->flags & (HF_MP_MASK | HF_TS_MASK)) ==
            (HF_MP_MASK | HF_TS_MASK)) {
            gen_exception(s, EXCP07_PREX, pc_start - s->cs_base);
        } else {
            gen_update_cc_op(s);
            gen_jmp_im(pc_start - s->cs_base);
            gen_helper_fwait(cpu_env);
        break;
    case 0xcc: /* int3 */
        gen_interrupt(s, EXCP03_INT3, pc_start - s->cs_base, s->pc - s->cs_base);
        break;
    case 0xcd: /* int N */
        val = cpu_ldub_code(env, s->pc++);
        if (s->vm86 && s->iopl != 3) {
            gen_exception(s, EXCP0D_GPF, pc_start - s->cs_base);
        } else {
            gen_interrupt(s, val, pc_start - s->cs_base, s->pc - s->cs_base);
        break;
    case 0xce: /* into */
        if (CODE64(s))
        gen_update_cc_op(s);
        gen_jmp_im(pc_start - s->cs_base);
        gen_helper_into(cpu_env, tcg_const_i32(s->pc - pc_start));
        break;
#ifdef WANT_ICEBP
    case 0xf1: /* icebp (undocumented, exits to external debugger) */
        gen_svm_check_intercept(s, pc_start, SVM_EXIT_ICEBP);
#if 1
        gen_debug(s, pc_start - s->cs_base);
#else
        /* start debug */
        tb_flush(env);
        qemu_set_log(CPU_LOG_INT | CPU_LOG_TB_IN_ASM);
#endif
        break;
#endif
    case 0xfa: /* cli */
        if (!s->vm86) {
            if (s->cpl <= s->iopl) {
                gen_helper_cli(cpu_env);
            } else {
                gen_exception(s, EXCP0D_GPF, pc_start - s->cs_base);
        } else {
            if (s->iopl == 3) {
                gen_helper_cli(cpu_env);
            } else {
                gen_exception(s, EXCP0D_GPF, pc_start - s->cs_base);
        break;
    case 0xfb: /* sti */
        if (!s->vm86) {
            if (s->cpl <= s->iopl) {
            gen_sti:
                gen_helper_sti(cpu_env);
                /* interruptions are enabled only the first insn after sti */
                /* If several instructions disable interrupts, only the
                   _first_ does it */
                if (!(s->tb->flags & HF_INHIBIT_IRQ_MASK))
                    gen_helper_set_inhibit_irq(cpu_env);
                /* give a chance to handle pending irqs */
                gen_jmp_im(s->pc - s->cs_base);
                gen_eob(s);
            } else {
                gen_exception(s, EXCP0D_GPF, pc_start - s->cs_base);
        } else {
            if (s->iopl == 3) {
                goto gen_sti;
            } else {
                gen_exception(s, EXCP0D_GPF, pc_start - s->cs_base);
        break;
    case 0x62: /* bound */
        if (CODE64(s))
        ot = dflag ? OT_LONG : OT_WORD;
        modrm = cpu_ldub_code(env, s->pc++);
        reg = (modrm >> 3) & 7;
        mod = (modrm >> 6) & 3;
        if (mod == 3)
        gen_op_mov_TN_reg(ot, 0, reg);
        gen_lea_modrm(env, s, modrm, &reg_addr, &offset_addr);
        gen_jmp_im(pc_start - s->cs_base);
        tcg_gen_trunc_tl_i32(cpu_tmp2_i32, cpu_T[0]);
        if (ot == OT_WORD) {
            gen_helper_boundw(cpu_env, cpu_A0, cpu_tmp2_i32);
        } else {
            gen_helper_boundl(cpu_env, cpu_A0, cpu_tmp2_i32);
        break;
    case 0x1c8 ... 0x1cf: /* bswap reg */
        reg = (b & 7) | REX_B(s);
#ifdef TARGET_X86_64
        if (dflag == 2) {
            gen_op_mov_TN_reg(OT_QUAD, 0, reg);
            tcg_gen_bswap64_i64(cpu_T[0], cpu_T[0]);
            gen_op_mov_reg_T0(OT_QUAD, reg);
        } else
#endif
        {
            gen_op_mov_TN_reg(OT_LONG, 0, reg);
            tcg_gen_ext32u_tl(cpu_T[0], cpu_T[0]);
            tcg_gen_bswap32_tl(cpu_T[0], cpu_T[0]);
            gen_op_mov_reg_T0(OT_LONG, reg);
        break;
    case 0xd6: /* salc */
        if (CODE64(s))
        gen_compute_eflags_c(s, cpu_T[0]);
        tcg_gen_neg_tl(cpu_T[0], cpu_T[0]);
        gen_op_mov_reg_T0(OT_BYTE, R_EAX);
        break;
    case 0xe0: /* loopnz */
    case 0xe1: /* loopz */
    case 0xe2: /* loop */
    case 0xe3: /* jecxz */
        {
            int l1, l2, l3;
            tval = (int8_t)insn_get(env, s, OT_BYTE);
            next_eip = s->pc - s->cs_base;
            tval += next_eip;
            if (s->dflag == 0)
                tval &= 0xffff;
            l1 = gen_new_label();
            l2 = gen_new_label();
            l3 = gen_new_label();
            b &= 3;
            switch(b) {
            case 0: /* loopnz */
            case 1: /* loopz */
                gen_op_add_reg_im(s->aflag, R_ECX, -1);
                gen_op_jz_ecx(s->aflag, l3);
                gen_jcc1(s, (JCC_Z << 1) | (b ^ 1), l1);
                break;
            case 2: /* loop */
                gen_op_add_reg_im(s->aflag, R_ECX, -1);
                gen_op_jnz_ecx(s->aflag, l1);
                break;
            default:
            case 3: /* jcxz */
                gen_op_jz_ecx(s->aflag, l1);
                break;
            gen_set_label(l3);
            gen_jmp_im(next_eip);
            tcg_gen_br(l2);
            gen_set_label(l1);
            gen_jmp_im(tval);
            gen_set_label(l2);
            gen_eob(s);
        break;
    case 0x130: /* wrmsr */
    case 0x132: /* rdmsr */
        if (s->cpl != 0) {
            gen_exception(s, EXCP0D_GPF, pc_start - s->cs_base);
        } else {
            gen_update_cc_op(s);
            gen_jmp_im(pc_start - s->cs_base);
            if (b & 2) {
                gen_helper_rdmsr(cpu_env);
            } else {
                gen_helper_wrmsr(cpu_env);
        break;
    case 0x131: /* rdtsc */
        gen_update_cc_op(s);
        gen_jmp_im(pc_start - s->cs_base);
        if (use_icount)
            gen_io_start();
        gen_helper_rdtsc(cpu_env);
        if (use_icount) {
            gen_io_end();
            gen_jmp(s, s->pc - s->cs_base);
        break;
    case 0x133: /* rdpmc */
        gen_update_cc_op(s);
        gen_jmp_im(pc_start - s->cs_base);
        gen_helper_rdpmc(cpu_env);
        break;
    case 0x134: /* sysenter */
        /* For Intel SYSENTER is valid on 64-bit */
        if (CODE64(s) && env->cpuid_vendor1 != CPUID_VENDOR_INTEL_1)
        if (!s->pe) {
            gen_exception(s, EXCP0D_GPF, pc_start - s->cs_base);
        } else {
            gen_update_cc_op(s);
            gen_jmp_im(pc_start - s->cs_base);
            gen_helper_sysenter(cpu_env);
            gen_eob(s);
        break;
    case 0x135: /* sysexit */
        /* For Intel SYSEXIT is valid on 64-bit */
        if (CODE64(s) && env->cpuid_vendor1 != CPUID_VENDOR_INTEL_1)
        if (!s->pe) {
            gen_exception(s, EXCP0D_GPF, pc_start - s->cs_base);
        } else {
            gen_update_cc_op(s);
            gen_jmp_im(pc_start - s->cs_base);
            gen_helper_sysexit(cpu_env, tcg_const_i32(dflag));
            gen_eob(s);
        break;
#ifdef TARGET_X86_64
    case 0x105: /* syscall */
        /* XXX: is it usable in real mode ? */
        gen_update_cc_op(s);
        gen_jmp_im(pc_start - s->cs_base);
        gen_helper_syscall(cpu_env, tcg_const_i32(s->pc - pc_start));
        gen_eob(s);
        break;
    case 0x107: /* sysret */
        if (!s->pe) {
            gen_exception(s, EXCP0D_GPF, pc_start - s->cs_base);
        } else {
            gen_update_cc_op(s);
            gen_jmp_im(pc_start - s->cs_base);
            gen_helper_sysret(cpu_env, tcg_const_i32(s->dflag));
            /* condition codes are modified only in long mode */
            if (s->lma) {
                set_cc_op(s, CC_OP_EFLAGS);
            gen_eob(s);
        break;
#endif
    case 0x1a2: /* cpuid */
        gen_update_cc_op(s);
        gen_jmp_im(pc_start - s->cs_base);
        gen_helper_cpuid(cpu_env);
        break;
    case 0xf4: /* hlt */
        if (s->cpl != 0) {
            gen_exception(s, EXCP0D_GPF, pc_start - s->cs_base);
        } else {
            gen_update_cc_op(s);
            gen_jmp_im(pc_start - s->cs_base);
            gen_helper_hlt(cpu_env, tcg_const_i32(s->pc - pc_start));
            s->is_jmp = DISAS_TB_JUMP;
        break;
    case 0x100:
        modrm = cpu_ldub_code(env, s->pc++);
        mod = (modrm >> 6) & 3;
        op = (modrm >> 3) & 7;
        switch(op) {
        case 0: /* sldt */
            if (!s->pe || s->vm86)
            gen_svm_check_intercept(s, pc_start, SVM_EXIT_LDTR_READ);
            tcg_gen_ld32u_tl(cpu_T[0], cpu_env, offsetof(CPUX86State,ldt.selector));
            ot = OT_WORD;
            if (mod == 3)
                ot += s->dflag;
            gen_ldst_modrm(env, s, modrm, ot, OR_TMP0, 1);
            break;
        case 2: /* lldt */
            if (!s->pe || s->vm86)
            if (s->cpl != 0) {
                gen_exception(s, EXCP0D_GPF, pc_start - s->cs_base);
            } else {
                gen_svm_check_intercept(s, pc_start, SVM_EXIT_LDTR_WRITE);
                gen_ldst_modrm(env, s, modrm, OT_WORD, OR_TMP0, 0);
                gen_jmp_im(pc_start - s->cs_base);
                tcg_gen_trunc_tl_i32(cpu_tmp2_i32, cpu_T[0]);
                gen_helper_lldt(cpu_env, cpu_tmp2_i32);
            break;
        case 1: /* str */
            if (!s->pe || s->vm86)
            gen_svm_check_intercept(s, pc_start, SVM_EXIT_TR_READ);
            tcg_gen_ld32u_tl(cpu_T[0], cpu_env, offsetof(CPUX86State,tr.selector));
            ot = OT_WORD;
            if (mod == 3)
                ot += s->dflag;
            gen_ldst_modrm(env, s, modrm, ot, OR_TMP0, 1);
            break;
        case 3: /* ltr */
            if (!s->pe || s->vm86)
            if (s->cpl != 0) {
                gen_exception(s, EXCP0D_GPF, pc_start - s->cs_base);
            } else {
                gen_svm_check_intercept(s, pc_start, SVM_EXIT_TR_WRITE);
                gen_ldst_modrm(env, s, modrm, OT_WORD, OR_TMP0, 0);
                gen_jmp_im(pc_start - s->cs_base);
                tcg_gen_trunc_tl_i32(cpu_tmp2_i32, cpu_T[0]);
                gen_helper_ltr(cpu_env, cpu_tmp2_i32);
            break;
        case 4: /* verr */
        case 5: /* verw */
            if (!s->pe || s->vm86)
            gen_ldst_modrm(env, s, modrm, OT_WORD, OR_TMP0, 0);
            gen_update_cc_op(s);
            if (op == 4) {
                gen_helper_verr(cpu_env, cpu_T[0]);
            } else {
                gen_helper_verw(cpu_env, cpu_T[0]);
            set_cc_op(s, CC_OP_EFLAGS);
            break;
        default:
        break;
    case 0x101:
        modrm = cpu_ldub_code(env, s->pc++);
        mod = (modrm >> 6) & 3;
        op = (modrm >> 3) & 7;
        rm = modrm & 7;
        switch(op) {
        case 0: /* sgdt */
            if (mod == 3)
            gen_svm_check_intercept(s, pc_start, SVM_EXIT_GDTR_READ);
            gen_lea_modrm(env, s, modrm, &reg_addr, &offset_addr);
            tcg_gen_ld32u_tl(cpu_T[0], cpu_env, offsetof(CPUX86State, gdt.limit));
            gen_op_st_T0_A0(OT_WORD + s->mem_index);
            gen_add_A0_im(s, 2);
            tcg_gen_ld_tl(cpu_T[0], cpu_env, offsetof(CPUX86State, gdt.base));
            if (!s->dflag)
                gen_op_andl_T0_im(0xffffff);
            gen_op_st_T0_A0(CODE64(s) + OT_LONG + s->mem_index);
            break;
        case 1:
            if (mod == 3) {
                switch (rm) {
                case 0: /* monitor */
                    if (!(s->cpuid_ext_features & CPUID_EXT_MONITOR) ||
                        s->cpl != 0)
                    gen_update_cc_op(s);
                    gen_jmp_im(pc_start - s->cs_base);
#ifdef TARGET_X86_64
                    if (s->aflag == 2) {
                        gen_op_movq_A0_reg(R_EAX);
                    } else
#endif
                    {
                        gen_op_movl_A0_reg(R_EAX);
                        if (s->aflag == 0)
                            gen_op_andl_A0_ffff();
                    gen_add_A0_ds_seg(s);
                    gen_helper_monitor(cpu_env, cpu_A0);
                    break;
                case 1: /* mwait */
                    if (!(s->cpuid_ext_features & CPUID_EXT_MONITOR) ||
                        s->cpl != 0)
                    gen_update_cc_op(s);
                    gen_jmp_im(pc_start - s->cs_base);
                    gen_helper_mwait(cpu_env, tcg_const_i32(s->pc - pc_start));
                    gen_eob(s);
                    break;
                case 2: /* clac */
                    if (!(s->cpuid_7_0_ebx_features & CPUID_7_0_EBX_SMAP) ||
                        s->cpl != 0) {
                    gen_helper_clac(cpu_env);
                    gen_jmp_im(s->pc - s->cs_base);
                    gen_eob(s);
                    break;
                case 3: /* stac */
                    if (!(s->cpuid_7_0_ebx_features & CPUID_7_0_EBX_SMAP) ||
                        s->cpl != 0) {
                    gen_helper_stac(cpu_env);
                    gen_jmp_im(s->pc - s->cs_base);
                    gen_eob(s);
                    break;
                default:
            } else { /* sidt */
                gen_svm_check_intercept(s, pc_start, SVM_EXIT_IDTR_READ);
                gen_lea_modrm(env, s, modrm, &reg_addr, &offset_addr);
                tcg_gen_ld32u_tl(cpu_T[0], cpu_env, offsetof(CPUX86State, idt.limit));
                gen_op_st_T0_A0(OT_WORD + s->mem_index);
                gen_add_A0_im(s, 2);
                tcg_gen_ld_tl(cpu_T[0], cpu_env, offsetof(CPUX86State, idt.base));
                if (!s->dflag)
                    gen_op_andl_T0_im(0xffffff);
                gen_op_st_T0_A0(CODE64(s) + OT_LONG + s->mem_index);
            break;
        case 2: /* lgdt */
        case 3: /* lidt */
            if (mod == 3) {
                gen_update_cc_op(s);
                gen_jmp_im(pc_start - s->cs_base);
                switch(rm) {
                case 0: /* VMRUN */
                    if (!(s->flags & HF_SVME_MASK) || !s->pe)
                    if (s->cpl != 0) {
                        gen_exception(s, EXCP0D_GPF, pc_start - s->cs_base);
                        break;
                    } else {
                        gen_helper_vmrun(cpu_env, tcg_const_i32(s->aflag),
                                         tcg_const_i32(s->pc - pc_start));
                        tcg_gen_exit_tb(0);
                        s->is_jmp = DISAS_TB_JUMP;
                    break;
                case 1: /* VMMCALL */
                    if (!(s->flags & HF_SVME_MASK))
                    gen_helper_vmmcall(cpu_env);
                    break;
                case 2: /* VMLOAD */
                    if (!(s->flags & HF_SVME_MASK) || !s->pe)
                    if (s->cpl != 0) {
                        gen_exception(s, EXCP0D_GPF, pc_start - s->cs_base);
                        break;
                    } else {
                        gen_helper_vmload(cpu_env, tcg_const_i32(s->aflag));
                    break;
                case 3: /* VMSAVE */
                    if (!(s->flags & HF_SVME_MASK) || !s->pe)
                    if (s->cpl != 0) {
                        gen_exception(s, EXCP0D_GPF, pc_start - s->cs_base);
                        break;
                    } else {
                        gen_helper_vmsave(cpu_env, tcg_const_i32(s->aflag));
                    break;
                case 4: /* STGI */
                    if ((!(s->flags & HF_SVME_MASK) &&
                         !(s->cpuid_ext3_features & CPUID_EXT3_SKINIT)) || 
                        !s->pe)
                    if (s->cpl != 0) {
                        gen_exception(s, EXCP0D_GPF, pc_start - s->cs_base);
                        break;
                    } else {
                        gen_helper_stgi(cpu_env);
                    break;
                case 5: /* CLGI */
                    if (!(s->flags & HF_SVME_MASK) || !s->pe)
                    if (s->cpl != 0) {
                        gen_exception(s, EXCP0D_GPF, pc_start - s->cs_base);
                        break;
                    } else {
                        gen_helper_clgi(cpu_env);
                    break;
                case 6: /* SKINIT */
                    if ((!(s->flags & HF_SVME_MASK) && 
                         !(s->cpuid_ext3_features & CPUID_EXT3_SKINIT)) || 
                        !s->pe)
                    gen_helper_skinit(cpu_env);
                    break;
                case 7: /* INVLPGA */
                    if (!(s->flags & HF_SVME_MASK) || !s->pe)
                    if (s->cpl != 0) {
                        gen_exception(s, EXCP0D_GPF, pc_start - s->cs_base);
                        break;
                    } else {
                        gen_helper_invlpga(cpu_env, tcg_const_i32(s->aflag));
                    break;
                default:
            } else if (s->cpl != 0) {
                gen_exception(s, EXCP0D_GPF, pc_start - s->cs_base);
            } else {
                gen_svm_check_intercept(s, pc_start,
                                        op==2 ? SVM_EXIT_GDTR_WRITE : SVM_EXIT_IDTR_WRITE);
                gen_lea_modrm(env, s, modrm, &reg_addr, &offset_addr);
                gen_op_ld_T1_A0(OT_WORD + s->mem_index);
                gen_add_A0_im(s, 2);
                gen_op_ld_T0_A0(CODE64(s) + OT_LONG + s->mem_index);
                if (!s->dflag)
                    gen_op_andl_T0_im(0xffffff);
                if (op == 2) {
                    tcg_gen_st_tl(cpu_T[0], cpu_env, offsetof(CPUX86State,gdt.base));
                    tcg_gen_st32_tl(cpu_T[1], cpu_env, offsetof(CPUX86State,gdt.limit));
                } else {
                    tcg_gen_st_tl(cpu_T[0], cpu_env, offsetof(CPUX86State,idt.base));
                    tcg_gen_st32_tl(cpu_T[1], cpu_env, offsetof(CPUX86State,idt.limit));
            break;
        case 4: /* smsw */
            gen_svm_check_intercept(s, pc_start, SVM_EXIT_READ_CR0);
#if defined TARGET_X86_64 && defined HOST_WORDS_BIGENDIAN
            tcg_gen_ld32u_tl(cpu_T[0], cpu_env, offsetof(CPUX86State,cr[0]) + 4);
#else
            tcg_gen_ld32u_tl(cpu_T[0], cpu_env, offsetof(CPUX86State,cr[0]));
#endif
            gen_ldst_modrm(env, s, modrm, OT_WORD, OR_TMP0, 1);
            break;
        case 6: /* lmsw */
            if (s->cpl != 0) {
                gen_exception(s, EXCP0D_GPF, pc_start - s->cs_base);
            } else {
                gen_svm_check_intercept(s, pc_start, SVM_EXIT_WRITE_CR0);
                gen_ldst_modrm(env, s, modrm, OT_WORD, OR_TMP0, 0);
                gen_helper_lmsw(cpu_env, cpu_T[0]);
                gen_jmp_im(s->pc - s->cs_base);
                gen_eob(s);
            break;
        case 7:
            if (mod != 3) { /* invlpg */
                if (s->cpl != 0) {
                    gen_exception(s, EXCP0D_GPF, pc_start - s->cs_base);
                } else {
                    gen_update_cc_op(s);
                    gen_jmp_im(pc_start - s->cs_base);
                    gen_lea_modrm(env, s, modrm, &reg_addr, &offset_addr);
                    gen_helper_invlpg(cpu_env, cpu_A0);
                    gen_jmp_im(s->pc - s->cs_base);
                    gen_eob(s);
            } else {
                switch (rm) {
                case 0: /* swapgs */
#ifdef TARGET_X86_64
                    if (CODE64(s)) {
                        if (s->cpl != 0) {
                            gen_exception(s, EXCP0D_GPF, pc_start - s->cs_base);
                        } else {
                            tcg_gen_ld_tl(cpu_T[0], cpu_env,
                                offsetof(CPUX86State,segs[R_GS].base));
                            tcg_gen_ld_tl(cpu_T[1], cpu_env,
                                offsetof(CPUX86State,kernelgsbase));
                            tcg_gen_st_tl(cpu_T[1], cpu_env,
                                offsetof(CPUX86State,segs[R_GS].base));
                            tcg_gen_st_tl(cpu_T[0], cpu_env,
                                offsetof(CPUX86State,kernelgsbase));
                    } else
#endif
                    {
                    break;
                case 1: /* rdtscp */
                    if (!(s->cpuid_ext2_features & CPUID_EXT2_RDTSCP))
                    gen_update_cc_op(s);
                    gen_jmp_im(pc_start - s->cs_base);
                    if (use_icount)
                        gen_io_start();
                    gen_helper_rdtscp(cpu_env);
                    if (use_icount) {
                        gen_io_end();
                        gen_jmp(s, s->pc - s->cs_base);
                    break;
                default:
            break;
        default:
        break;
    case 0x108: /* invd */
    case 0x109: /* wbinvd */
        if (s->cpl != 0) {
            gen_exception(s, EXCP0D_GPF, pc_start - s->cs_base);
        } else {
            gen_svm_check_intercept(s, pc_start, (b & 2) ? SVM_EXIT_INVD : SVM_EXIT_WBINVD);
            /* nothing to do */
        break;
    case 0x63: /* arpl or movslS (x86_64) */
#ifdef TARGET_X86_64
        if (CODE64(s)) {
            int d_ot;
            /* d_ot is the size of destination */
            d_ot = dflag + OT_WORD;
            modrm = cpu_ldub_code(env, s->pc++);
            reg = ((modrm >> 3) & 7) | rex_r;
            mod = (modrm >> 6) & 3;
            rm = (modrm & 7) | REX_B(s);
            if (mod == 3) {
                gen_op_mov_TN_reg(OT_LONG, 0, rm);
                /* sign extend */
                if (d_ot == OT_QUAD)
                    tcg_gen_ext32s_tl(cpu_T[0], cpu_T[0]);
                gen_op_mov_reg_T0(d_ot, reg);
            } else {
                gen_lea_modrm(env, s, modrm, &reg_addr, &offset_addr);
                if (d_ot == OT_QUAD) {
                    gen_op_lds_T0_A0(OT_LONG + s->mem_index);
                } else {
                    gen_op_ld_T0_A0(OT_LONG + s->mem_index);
                gen_op_mov_reg_T0(d_ot, reg);
        } else
#endif
        {
            int label1;
            TCGv t0, t1, t2, a0;
            if (!s->pe || s->vm86)
            t0 = tcg_temp_local_new();
            t1 = tcg_temp_local_new();
            t2 = tcg_temp_local_new();
            ot = OT_WORD;
            modrm = cpu_ldub_code(env, s->pc++);
            reg = (modrm >> 3) & 7;
            mod = (modrm >> 6) & 3;
            rm = modrm & 7;
            if (mod != 3) {
                gen_lea_modrm(env, s, modrm, &reg_addr, &offset_addr);
                gen_op_ld_v(ot + s->mem_index, t0, cpu_A0);
                a0 = tcg_temp_local_new();
                tcg_gen_mov_tl(a0, cpu_A0);
            } else {
                gen_op_mov_v_reg(ot, t0, rm);
                TCGV_UNUSED(a0);
            gen_op_mov_v_reg(ot, t1, reg);
            tcg_gen_andi_tl(cpu_tmp0, t0, 3);
            tcg_gen_andi_tl(t1, t1, 3);
            tcg_gen_movi_tl(t2, 0);
            label1 = gen_new_label();
            tcg_gen_brcond_tl(TCG_COND_GE, cpu_tmp0, t1, label1);
            tcg_gen_andi_tl(t0, t0, ~3);
            tcg_gen_or_tl(t0, t0, t1);
            tcg_gen_movi_tl(t2, CC_Z);
            gen_set_label(label1);
            if (mod != 3) {
                gen_op_st_v(ot + s->mem_index, t0, a0);
                tcg_temp_free(a0);
           } else {
                gen_op_mov_reg_v(ot, rm, t0);
            gen_compute_eflags(s);
            tcg_gen_andi_tl(cpu_cc_src, cpu_cc_src, ~CC_Z);
            tcg_gen_or_tl(cpu_cc_src, cpu_cc_src, t2);
            tcg_temp_free(t0);
            tcg_temp_free(t1);
            tcg_temp_free(t2);
        break;
    case 0x102: /* lar */
    case 0x103: /* lsl */
        {
            int label1;
            TCGv t0;
            if (!s->pe || s->vm86)
            ot = dflag ? OT_LONG : OT_WORD;
            modrm = cpu_ldub_code(env, s->pc++);
            reg = ((modrm >> 3) & 7) | rex_r;
            gen_ldst_modrm(env, s, modrm, OT_WORD, OR_TMP0, 0);
            t0 = tcg_temp_local_new();
            gen_update_cc_op(s);
            if (b == 0x102) {
                gen_helper_lar(t0, cpu_env, cpu_T[0]);
            } else {
                gen_helper_lsl(t0, cpu_env, cpu_T[0]);
            tcg_gen_andi_tl(cpu_tmp0, cpu_cc_src, CC_Z);
            label1 = gen_new_label();
            tcg_gen_brcondi_tl(TCG_COND_EQ, cpu_tmp0, 0, label1);
            gen_op_mov_reg_v(ot, reg, t0);
            gen_set_label(label1);
            set_cc_op(s, CC_OP_EFLAGS);
            tcg_temp_free(t0);
        break;
    case 0x118:
        modrm = cpu_ldub_code(env, s->pc++);
        mod = (modrm >> 6) & 3;
        op = (modrm >> 3) & 7;
        switch(op) {
        case 0: /* prefetchnta */
        case 1: /* prefetchnt0 */
        case 2: /* prefetchnt0 */
        case 3: /* prefetchnt0 */
            if (mod == 3)
            gen_lea_modrm(env, s, modrm, &reg_addr, &offset_addr);
            /* nothing more to do */
            break;
        default: /* nop (multi byte) */
            gen_nop_modrm(env, s, modrm);
            break;
        break;
    case 0x119 ... 0x11f: /* nop (multi byte) */
        modrm = cpu_ldub_code(env, s->pc++);
        gen_nop_modrm(env, s, modrm);
        break;
    case 0x120: /* mov reg, crN */
    case 0x122: /* mov crN, reg */
        if (s->cpl != 0) {
            gen_exception(s, EXCP0D_GPF, pc_start - s->cs_base);
        } else {
            modrm = cpu_ldub_code(env, s->pc++);
            /* Ignore the mod bits (assume (modrm&0xc0)==0xc0).
             * AMD documentation (24594.pdf) and testing of
             * intel 386 and 486 processors all show that the mod bits
             * are assumed to be 1's, regardless of actual values.
             */
            rm = (modrm & 7) | REX_B(s);
            reg = ((modrm >> 3) & 7) | rex_r;
            if (CODE64(s))
                ot = OT_QUAD;
            else
                ot = OT_LONG;
            if ((prefixes & PREFIX_LOCK) && (reg == 0) &&
                (s->cpuid_ext3_features & CPUID_EXT3_CR8LEG)) {
                reg = 8;
            switch(reg) {
            case 0:
            case 2:
            case 3:
            case 4:
            case 8:
                gen_update_cc_op(s);
                gen_jmp_im(pc_start - s->cs_base);
                if (b & 2) {
                    gen_op_mov_TN_reg(ot, 0, rm);
                    gen_helper_write_crN(cpu_env, tcg_const_i32(reg),
                                         cpu_T[0]);
                    gen_jmp_im(s->pc - s->cs_base);
                    gen_eob(s);
                } else {
                    gen_helper_read_crN(cpu_T[0], cpu_env, tcg_const_i32(reg));
                    gen_op_mov_reg_T0(ot, rm);
                break;
            default:
        break;
    case 0x121: /* mov reg, drN */
    case 0x123: /* mov drN, reg */
        if (s->cpl != 0) {
            gen_exception(s, EXCP0D_GPF, pc_start - s->cs_base);
        } else {
            modrm = cpu_ldub_code(env, s->pc++);
            /* Ignore the mod bits (assume (modrm&0xc0)==0xc0).
             * AMD documentation (24594.pdf) and testing of
             * intel 386 and 486 processors all show that the mod bits
             * are assumed to be 1's, regardless of actual values.
             */
            rm = (modrm & 7) | REX_B(s);
            reg = ((modrm >> 3) & 7) | rex_r;
            if (CODE64(s))
                ot = OT_QUAD;
            else
                ot = OT_LONG;
            /* XXX: do it dynamically with CR4.DE bit */
            if (reg == 4 || reg == 5 || reg >= 8)
            if (b & 2) {
                gen_svm_check_intercept(s, pc_start, SVM_EXIT_WRITE_DR0 + reg);
                gen_op_mov_TN_reg(ot, 0, rm);
                gen_helper_movl_drN_T0(cpu_env, tcg_const_i32(reg), cpu_T[0]);
                gen_jmp_im(s->pc - s->cs_base);
                gen_eob(s);
            } else {
                gen_svm_check_intercept(s, pc_start, SVM_EXIT_READ_DR0 + reg);
                tcg_gen_ld_tl(cpu_T[0], cpu_env, offsetof(CPUX86State,dr[reg]));
                gen_op_mov_reg_T0(ot, rm);
        break;
    case 0x106: /* clts */
        if (s->cpl != 0) {
            gen_exception(s, EXCP0D_GPF, pc_start - s->cs_base);
        } else {
            gen_svm_check_intercept(s, pc_start, SVM_EXIT_WRITE_CR0);
            gen_helper_clts(cpu_env);
            /* abort block because static cpu state changed */
            gen_jmp_im(s->pc - s->cs_base);
            gen_eob(s);
        break;
    /* MMX/3DNow!/SSE/SSE2/SSE3/SSSE3/SSE4 support */
    case 0x1c3: /* MOVNTI reg, mem */
        if (!(s->cpuid_features & CPUID_SSE2))
        ot = s->dflag == 2 ? OT_QUAD : OT_LONG;
        modrm = cpu_ldub_code(env, s->pc++);
        mod = (modrm >> 6) & 3;
        if (mod == 3)
        reg = ((modrm >> 3) & 7) | rex_r;
        /* generate a generic store */
        gen_ldst_modrm(env, s, modrm, ot, reg, 1);
        break;
    case 0x1ae:
        modrm = cpu_ldub_code(env, s->pc++);
        mod = (modrm >> 6) & 3;
        op = (modrm >> 3) & 7;
        switch(op) {
        case 0: /* fxsave */
            if (mod == 3 || !(s->cpuid_features & CPUID_FXSR) ||
                (s->prefix & PREFIX_LOCK))
            if ((s->flags & HF_EM_MASK) || (s->flags & HF_TS_MASK)) {
                gen_exception(s, EXCP07_PREX, pc_start - s->cs_base);
                break;
            gen_lea_modrm(env, s, modrm, &reg_addr, &offset_addr);
            gen_update_cc_op(s);
            gen_jmp_im(pc_start - s->cs_base);
            gen_helper_fxsave(cpu_env, cpu_A0, tcg_const_i32((s->dflag == 2)));
            break;
        case 1: /* fxrstor */
            if (mod == 3 || !(s->cpuid_features & CPUID_FXSR) ||
                (s->prefix & PREFIX_LOCK))
            if ((s->flags & HF_EM_MASK) || (s->flags & HF_TS_MASK)) {
                gen_exception(s, EXCP07_PREX, pc_start - s->cs_base);
                break;
            gen_lea_modrm(env, s, modrm, &reg_addr, &offset_addr);
            gen_update_cc_op(s);
            gen_jmp_im(pc_start - s->cs_base);
            gen_helper_fxrstor(cpu_env, cpu_A0,
                               tcg_const_i32((s->dflag == 2)));
            break;
        case 2: /* ldmxcsr */
        case 3: /* stmxcsr */
            if (s->flags & HF_TS_MASK) {
                gen_exception(s, EXCP07_PREX, pc_start - s->cs_base);
                break;
            if ((s->flags & HF_EM_MASK) || !(s->flags & HF_OSFXSR_MASK) ||
                mod == 3)
            gen_lea_modrm(env, s, modrm, &reg_addr, &offset_addr);
            if (op == 2) {
                gen_op_ld_T0_A0(OT_LONG + s->mem_index);
                tcg_gen_trunc_tl_i32(cpu_tmp2_i32, cpu_T[0]);
                gen_helper_ldmxcsr(cpu_env, cpu_tmp2_i32);
            } else {
                tcg_gen_ld32u_tl(cpu_T[0], cpu_env, offsetof(CPUX86State, mxcsr));
                gen_op_st_T0_A0(OT_LONG + s->mem_index);
            break;
        case 5: /* lfence */
        case 6: /* mfence */
            if ((modrm & 0xc7) != 0xc0 || !(s->cpuid_features & CPUID_SSE2))
            break;
        case 7: /* sfence / clflush */
            if ((modrm & 0xc7) == 0xc0) {
                /* sfence */
                /* XXX: also check for cpuid_ext2_features & CPUID_EXT2_EMMX */
                if (!(s->cpuid_features & CPUID_SSE))
            } else {
                /* clflush */
                if (!(s->cpuid_features & CPUID_CLFLUSH))
                gen_lea_modrm(env, s, modrm, &reg_addr, &offset_addr);
            break;
        default:
        break;
    case 0x10d: /* 3DNow! prefetch(w) */
        modrm = cpu_ldub_code(env, s->pc++);
        mod = (modrm >> 6) & 3;
        if (mod == 3)
        gen_lea_modrm(env, s, modrm, &reg_addr, &offset_addr);
        /* ignore for now */
        break;
    case 0x1aa: /* rsm */
        gen_svm_check_intercept(s, pc_start, SVM_EXIT_RSM);
        if (!(s->flags & HF_SMM_MASK))
        gen_update_cc_op(s);
        gen_jmp_im(s->pc - s->cs_base);
        gen_helper_rsm(cpu_env);
        gen_eob(s);
        break;
    case 0x1b8: /* SSE4.2 popcnt */
        if ((prefixes & (PREFIX_REPZ | PREFIX_LOCK | PREFIX_REPNZ)) !=
             PREFIX_REPZ)
        if (!(s->cpuid_ext_features & CPUID_EXT_POPCNT))
        modrm = cpu_ldub_code(env, s->pc++);
        reg = ((modrm >> 3) & 7) | rex_r;
        if (s->prefix & PREFIX_DATA)
            ot = OT_WORD;
        else if (s->dflag != 2)
            ot = OT_LONG;
        else
            ot = OT_QUAD;
        gen_ldst_modrm(env, s, modrm, ot, OR_TMP0, 0);
        gen_helper_popcnt(cpu_T[0], cpu_env, cpu_T[0], tcg_const_i32(ot));
        gen_op_mov_reg_T0(ot, reg);
        set_cc_op(s, CC_OP_EFLAGS);
        break;
    case 0x10e ... 0x10f:
        /* 3DNow! instructions, ignore prefixes */
        s->prefix &= ~(PREFIX_REPZ | PREFIX_REPNZ | PREFIX_DATA);
    case 0x110 ... 0x117:
    case 0x128 ... 0x12f:
    case 0x138 ... 0x13a:
    case 0x150 ... 0x179:
    case 0x17c ... 0x17f:
    case 0x1c2:
    case 0x1c4 ... 0x1c6:
    case 0x1d0 ... 0x1fe:
        gen_sse(env, s, b, pc_start, rex_r);
        break;
    default:
    /* lock generation */
    if (s->prefix & PREFIX_LOCK)
        gen_helper_unlock();
    return s->pc;
 illegal_op:
    if (s->prefix & PREFIX_LOCK)
        gen_helper_unlock();
    /* XXX: ensure that no lock was generated */
    gen_exception(s, EXCP06_ILLOP, pc_start - s->cs_base);
    return s->pc;