static int decode_opc(MoxieCPU *cpu, DisasContext *ctx)

{

    CPUMoxieState *env = &cpu->env;



    /* Local cache for the instruction opcode.  */

    int opcode;

    /* Set the default instruction length.  */

    int length = 2;



    if (unlikely(qemu_loglevel_mask(CPU_LOG_TB_OP | CPU_LOG_TB_OP_OPT))) {

        tcg_gen_debug_insn_start(ctx->pc);

    }



    /* Examine the 16-bit opcode.  */

    opcode = ctx->opcode;



    /* Decode instruction.  */

    if (opcode & (1 << 15)) {

        if (opcode & (1 << 14)) {

            /* This is a Form 3 instruction.  */

            int inst = (opcode >> 10 & 0xf);



#define BRANCH(cond)                                                         \

    do {                                                                     \

        int l1 = gen_new_label();                                            \

        tcg_gen_brcond_i32(cond, cc_a, cc_b, l1);                            \

        gen_goto_tb(env, ctx, 1, ctx->pc+2);                                 \

        gen_set_label(l1);                                                   \

        gen_goto_tb(env, ctx, 0, extract_branch_offset(opcode) + ctx->pc+2); \

        ctx->bstate = BS_BRANCH;                                             \

    } while (0)



            switch (inst) {

            case 0x00: /* beq */

                BRANCH(TCG_COND_EQ);

                break;

            case 0x01: /* bne */

                BRANCH(TCG_COND_NE);

                break;

            case 0x02: /* blt */

                BRANCH(TCG_COND_LT);

                break;

            case 0x03: /* bgt */

                BRANCH(TCG_COND_GT);

                break;

            case 0x04: /* bltu */

                BRANCH(TCG_COND_LTU);

                break;

            case 0x05: /* bgtu */

                BRANCH(TCG_COND_GTU);

                break;

            case 0x06: /* bge */

                BRANCH(TCG_COND_GE);

                break;

            case 0x07: /* ble */

                BRANCH(TCG_COND_LE);

                break;

            case 0x08: /* bgeu */

                BRANCH(TCG_COND_GEU);

                break;

            case 0x09: /* bleu */

                BRANCH(TCG_COND_LEU);

                break;

            default:

                {

                    TCGv temp = tcg_temp_new_i32();

                    tcg_gen_movi_i32(cpu_pc, ctx->pc);

                    tcg_gen_movi_i32(temp, MOXIE_EX_BAD);

                    gen_helper_raise_exception(cpu_env, temp);

                    tcg_temp_free_i32(temp);

                }

                break;

            }

        } else {

            /* This is a Form 2 instruction.  */

            int inst = (opcode >> 12 & 0x3);

            switch (inst) {

            case 0x00: /* inc */

                {

                    int a = (opcode >> 8) & 0xf;

                    unsigned int v = (opcode & 0xff);

                    tcg_gen_addi_i32(REG(a), REG(a), v);

                }

                break;

            case 0x01: /* dec */

                {

                    int a = (opcode >> 8) & 0xf;

                    unsigned int v = (opcode & 0xff);

                    tcg_gen_subi_i32(REG(a), REG(a), v);

                }

                break;

            case 0x02: /* gsr */

                {

                    int a = (opcode >> 8) & 0xf;

                    unsigned v = (opcode & 0xff);

                    tcg_gen_ld_i32(REG(a), cpu_env,

                                   offsetof(CPUMoxieState, sregs[v]));

                }

                break;

            case 0x03: /* ssr */

                {

                    int a = (opcode >> 8) & 0xf;

                    unsigned v = (opcode & 0xff);

                    tcg_gen_st_i32(REG(a), cpu_env,

                                   offsetof(CPUMoxieState, sregs[v]));

                }

                break;

            default:

                {

                    TCGv temp = tcg_temp_new_i32();

                    tcg_gen_movi_i32(cpu_pc, ctx->pc);

                    tcg_gen_movi_i32(temp, MOXIE_EX_BAD);

                    gen_helper_raise_exception(cpu_env, temp);

                    tcg_temp_free_i32(temp);

                }

                break;

            }

        }

    } else {

        /* This is a Form 1 instruction.  */

        int inst = opcode >> 8;

        switch (inst) {

        case 0x00: /* nop */

            break;

        case 0x01: /* ldi.l (immediate) */

            {

                int reg = (opcode >> 4) & 0xf;

                int val = cpu_ldl_code(env, ctx->pc+2);

                tcg_gen_movi_i32(REG(reg), val);

                length = 6;

            }

            break;

        case 0x02: /* mov (register-to-register) */

            {

                int dest  = (opcode >> 4) & 0xf;

                int src = opcode & 0xf;

                tcg_gen_mov_i32(REG(dest), REG(src));

            }

            break;

        case 0x03: /* jsra */

            {

                TCGv t1 = tcg_temp_new_i32();

                TCGv t2 = tcg_temp_new_i32();



                tcg_gen_movi_i32(t1, ctx->pc + 6);



                /* Make space for the static chain and return address.  */

                tcg_gen_subi_i32(t2, REG(1), 8);

                tcg_gen_mov_i32(REG(1), t2);

                tcg_gen_qemu_st32(t1, REG(1), ctx->memidx);



                /* Push the current frame pointer.  */

                tcg_gen_subi_i32(t2, REG(1), 4);

                tcg_gen_mov_i32(REG(1), t2);

                tcg_gen_qemu_st32(REG(0), REG(1), ctx->memidx);



                /* Set the pc and $fp.  */

                tcg_gen_mov_i32(REG(0), REG(1));



                gen_goto_tb(env, ctx, 0, cpu_ldl_code(env, ctx->pc+2));



                tcg_temp_free_i32(t1);

                tcg_temp_free_i32(t2);



                ctx->bstate = BS_BRANCH;

                length = 6;

            }

            break;

        case 0x04: /* ret */

            {

                TCGv t1 = tcg_temp_new_i32();



                /* The new $sp is the old $fp.  */

                tcg_gen_mov_i32(REG(1), REG(0));



                /* Pop the frame pointer.  */

                tcg_gen_qemu_ld32u(REG(0), REG(1), ctx->memidx);

                tcg_gen_addi_i32(t1, REG(1), 4);

                tcg_gen_mov_i32(REG(1), t1);





                /* Pop the return address and skip over the static chain

                   slot.  */

                tcg_gen_qemu_ld32u(cpu_pc, REG(1), ctx->memidx);

                tcg_gen_addi_i32(t1, REG(1), 8);

                tcg_gen_mov_i32(REG(1), t1);



                tcg_temp_free_i32(t1);



                /* Jump... */

                tcg_gen_exit_tb(0);



                ctx->bstate = BS_BRANCH;

            }

            break;

        case 0x05: /* add.l */

            {

                int a = (opcode >> 4) & 0xf;

                int b = opcode & 0xf;



                tcg_gen_add_i32(REG(a), REG(a), REG(b));

            }

            break;

        case 0x06: /* push */

            {

                int a = (opcode >> 4) & 0xf;

                int b = opcode & 0xf;



                TCGv t1 = tcg_temp_new_i32();

                tcg_gen_subi_i32(t1, REG(a), 4);

                tcg_gen_mov_i32(REG(a), t1);

                tcg_gen_qemu_st32(REG(b), REG(a), ctx->memidx);

                tcg_temp_free_i32(t1);

            }

            break;

        case 0x07: /* pop */

            {

                int a = (opcode >> 4) & 0xf;

                int b = opcode & 0xf;

                TCGv t1 = tcg_temp_new_i32();



                tcg_gen_qemu_ld32u(REG(b), REG(a), ctx->memidx);

                tcg_gen_addi_i32(t1, REG(a), 4);

                tcg_gen_mov_i32(REG(a), t1);

                tcg_temp_free_i32(t1);

            }

            break;

        case 0x08: /* lda.l */

            {

                int reg = (opcode >> 4) & 0xf;



                TCGv ptr = tcg_temp_new_i32();

                tcg_gen_movi_i32(ptr, cpu_ldl_code(env, ctx->pc+2));

                tcg_gen_qemu_ld32u(REG(reg), ptr, ctx->memidx);

                tcg_temp_free_i32(ptr);



                length = 6;

            }

            break;

        case 0x09: /* sta.l */

            {

                int val = (opcode >> 4) & 0xf;



                TCGv ptr = tcg_temp_new_i32();

                tcg_gen_movi_i32(ptr, cpu_ldl_code(env, ctx->pc+2));

                tcg_gen_qemu_st32(REG(val), ptr, ctx->memidx);

                tcg_temp_free_i32(ptr);



                length = 6;

            }

            break;

        case 0x0a: /* ld.l (register indirect) */

            {

                int src  = opcode & 0xf;

                int dest = (opcode >> 4) & 0xf;



                tcg_gen_qemu_ld32u(REG(dest), REG(src), ctx->memidx);

            }

            break;

        case 0x0b: /* st.l */

            {

                int dest = (opcode >> 4) & 0xf;

                int val  = opcode & 0xf;



                tcg_gen_qemu_st32(REG(val), REG(dest), ctx->memidx);

            }

            break;

        case 0x0c: /* ldo.l */

            {

                int a = (opcode >> 4) & 0xf;

                int b = opcode & 0xf;



                TCGv t1 = tcg_temp_new_i32();

                TCGv t2 = tcg_temp_new_i32();

                tcg_gen_addi_i32(t1, REG(b), cpu_ldl_code(env, ctx->pc+2));

                tcg_gen_qemu_ld32u(t2, t1, ctx->memidx);

                tcg_gen_mov_i32(REG(a), t2);



                tcg_temp_free_i32(t1);

                tcg_temp_free_i32(t2);



                length = 6;

            }

            break;

        case 0x0d: /* sto.l */

            {

                int a = (opcode >> 4) & 0xf;

                int b = opcode & 0xf;



                TCGv t1 = tcg_temp_new_i32();

                TCGv t2 = tcg_temp_new_i32();

                tcg_gen_addi_i32(t1, REG(a), cpu_ldl_code(env, ctx->pc+2));

                tcg_gen_qemu_st32(REG(b), t1, ctx->memidx);



                tcg_temp_free_i32(t1);

                tcg_temp_free_i32(t2);



                length = 6;

            }

            break;

        case 0x0e: /* cmp */

            {

                int a  = (opcode >> 4) & 0xf;

                int b  = opcode & 0xf;



                tcg_gen_mov_i32(cc_a, REG(a));

                tcg_gen_mov_i32(cc_b, REG(b));

            }

            break;

        case 0x19: /* jsr */

            {

                int fnreg = (opcode >> 4) & 0xf;



                /* Load the stack pointer into T0.  */

                TCGv t1 = tcg_temp_new_i32();

                TCGv t2 = tcg_temp_new_i32();



                tcg_gen_movi_i32(t1, ctx->pc+2);



                /* Make space for the static chain and return address.  */

                tcg_gen_subi_i32(t2, REG(1), 8);

                tcg_gen_mov_i32(REG(1), t2);

                tcg_gen_qemu_st32(t1, REG(1), ctx->memidx);



                /* Push the current frame pointer.  */

                tcg_gen_subi_i32(t2, REG(1), 4);

                tcg_gen_mov_i32(REG(1), t2);

                tcg_gen_qemu_st32(REG(0), REG(1), ctx->memidx);



                /* Set the pc and $fp.  */

                tcg_gen_mov_i32(REG(0), REG(1));

                tcg_gen_mov_i32(cpu_pc, REG(fnreg));

                tcg_temp_free_i32(t1);

                tcg_temp_free_i32(t2);

                tcg_gen_exit_tb(0);

                ctx->bstate = BS_BRANCH;

            }

            break;

        case 0x1a: /* jmpa */

            {

                tcg_gen_movi_i32(cpu_pc, cpu_ldl_code(env, ctx->pc+2));

                tcg_gen_exit_tb(0);

                ctx->bstate = BS_BRANCH;

                length = 6;

            }

            break;

        case 0x1b: /* ldi.b (immediate) */

            {

                int reg = (opcode >> 4) & 0xf;

                int val = cpu_ldl_code(env, ctx->pc+2);

                tcg_gen_movi_i32(REG(reg), val);

                length = 6;

            }

            break;

        case 0x1c: /* ld.b (register indirect) */

            {

                int src  = opcode & 0xf;

                int dest = (opcode >> 4) & 0xf;



                tcg_gen_qemu_ld8u(REG(dest), REG(src), ctx->memidx);

            }

            break;

        case 0x1d: /* lda.b */

            {

                int reg = (opcode >> 4) & 0xf;



                TCGv ptr = tcg_temp_new_i32();

                tcg_gen_movi_i32(ptr, cpu_ldl_code(env, ctx->pc+2));

                tcg_gen_qemu_ld8u(REG(reg), ptr, ctx->memidx);

                tcg_temp_free_i32(ptr);



                length = 6;

            }

            break;

        case 0x1e: /* st.b */

            {

                int dest = (opcode >> 4) & 0xf;

                int val  = opcode & 0xf;



                tcg_gen_qemu_st8(REG(val), REG(dest), ctx->memidx);

            }

            break;

        case 0x1f: /* sta.b */

            {

                int val = (opcode >> 4) & 0xf;



                TCGv ptr = tcg_temp_new_i32();

                tcg_gen_movi_i32(ptr, cpu_ldl_code(env, ctx->pc+2));

                tcg_gen_qemu_st8(REG(val), ptr, ctx->memidx);

                tcg_temp_free_i32(ptr);



                length = 6;

            }

            break;

        case 0x20: /* ldi.s (immediate) */

            {

                int reg = (opcode >> 4) & 0xf;

                int val = cpu_ldl_code(env, ctx->pc+2);

                tcg_gen_movi_i32(REG(reg), val);

                length = 6;

            }

            break;

        case 0x21: /* ld.s (register indirect) */

            {

                int src  = opcode & 0xf;

                int dest = (opcode >> 4) & 0xf;



                tcg_gen_qemu_ld16u(REG(dest), REG(src), ctx->memidx);

            }

            break;

        case 0x22: /* lda.s */

            {

                int reg = (opcode >> 4) & 0xf;



                TCGv ptr = tcg_temp_new_i32();

                tcg_gen_movi_i32(ptr, cpu_ldl_code(env, ctx->pc+2));

                tcg_gen_qemu_ld16u(REG(reg), ptr, ctx->memidx);

                tcg_temp_free_i32(ptr);



                length = 6;

            }

            break;

        case 0x23: /* st.s */

            {

                int dest = (opcode >> 4) & 0xf;

                int val  = opcode & 0xf;



                tcg_gen_qemu_st16(REG(val), REG(dest), ctx->memidx);

            }

            break;

        case 0x24: /* sta.s */

            {

                int val = (opcode >> 4) & 0xf;



                TCGv ptr = tcg_temp_new_i32();

                tcg_gen_movi_i32(ptr, cpu_ldl_code(env, ctx->pc+2));

                tcg_gen_qemu_st16(REG(val), ptr, ctx->memidx);

                tcg_temp_free_i32(ptr);



                length = 6;

            }

            break;

        case 0x25: /* jmp */

            {

                int reg = (opcode >> 4) & 0xf;

                tcg_gen_mov_i32(cpu_pc, REG(reg));

                tcg_gen_exit_tb(0);

                ctx->bstate = BS_BRANCH;

            }

            break;

        case 0x26: /* and */

            {

                int a = (opcode >> 4) & 0xf;

                int b = opcode & 0xf;



                tcg_gen_and_i32(REG(a), REG(a), REG(b));

            }

            break;

        case 0x27: /* lshr */

            {

                int a = (opcode >> 4) & 0xf;

                int b = opcode & 0xf;



                TCGv sv = tcg_temp_new_i32();

                tcg_gen_andi_i32(sv, REG(b), 0x1f);

                tcg_gen_shr_i32(REG(a), REG(a), sv);

                tcg_temp_free_i32(sv);

            }

            break;

        case 0x28: /* ashl */

            {

                int a = (opcode >> 4) & 0xf;

                int b = opcode & 0xf;



                TCGv sv = tcg_temp_new_i32();

                tcg_gen_andi_i32(sv, REG(b), 0x1f);

                tcg_gen_shl_i32(REG(a), REG(a), sv);

                tcg_temp_free_i32(sv);

            }

            break;

        case 0x29: /* sub.l */

            {

                int a = (opcode >> 4) & 0xf;

                int b = opcode & 0xf;



                tcg_gen_sub_i32(REG(a), REG(a), REG(b));

            }

            break;

        case 0x2a: /* neg */

            {

                int a = (opcode >> 4) & 0xf;

                int b = opcode & 0xf;



                tcg_gen_neg_i32(REG(a), REG(b));

            }

            break;

        case 0x2b: /* or */

            {

                int a = (opcode >> 4) & 0xf;

                int b = opcode & 0xf;



                tcg_gen_or_i32(REG(a), REG(a), REG(b));

            }

            break;

        case 0x2c: /* not */

            {

                int a = (opcode >> 4) & 0xf;

                int b = opcode & 0xf;



                tcg_gen_not_i32(REG(a), REG(b));

            }

            break;

        case 0x2d: /* ashr */

            {

                int a = (opcode >> 4) & 0xf;

                int b = opcode & 0xf;



                TCGv sv = tcg_temp_new_i32();

                tcg_gen_andi_i32(sv, REG(b), 0x1f);

                tcg_gen_sar_i32(REG(a), REG(a), sv);

                tcg_temp_free_i32(sv);

            }

            break;

        case 0x2e: /* xor */

            {

                int a = (opcode >> 4) & 0xf;

                int b = opcode & 0xf;



                tcg_gen_xor_i32(REG(a), REG(a), REG(b));

            }

            break;

        case 0x2f: /* mul.l */

            {

                int a = (opcode >> 4) & 0xf;

                int b = opcode & 0xf;



                tcg_gen_mul_i32(REG(a), REG(a), REG(b));

            }

            break;

        case 0x30: /* swi */

            {

                int val = cpu_ldl_code(env, ctx->pc+2);



                TCGv temp = tcg_temp_new_i32();

                tcg_gen_movi_i32(temp, val);

                tcg_gen_st_i32(temp, cpu_env,

                               offsetof(CPUMoxieState, sregs[3]));

                tcg_gen_movi_i32(cpu_pc, ctx->pc);

                tcg_gen_movi_i32(temp, MOXIE_EX_SWI);

                gen_helper_raise_exception(cpu_env, temp);

                tcg_temp_free_i32(temp);



                length = 6;

            }

            break;

        case 0x31: /* div.l */

            {

                int a = (opcode >> 4) & 0xf;

                int b = opcode & 0xf;

                tcg_gen_movi_i32(cpu_pc, ctx->pc);

                gen_helper_div(REG(a), cpu_env, REG(a), REG(b));

            }

            break;

        case 0x32: /* udiv.l */

            {

                int a = (opcode >> 4) & 0xf;

                int b = opcode & 0xf;

                tcg_gen_movi_i32(cpu_pc, ctx->pc);

                gen_helper_udiv(REG(a), cpu_env, REG(a), REG(b));

            }

            break;

        case 0x33: /* mod.l */

            {

                int a = (opcode >> 4) & 0xf;

                int b = opcode & 0xf;

                tcg_gen_rem_i32(REG(a), REG(a), REG(b));

            }

            break;

        case 0x34: /* umod.l */

            {

                int a = (opcode >> 4) & 0xf;

                int b = opcode & 0xf;

                tcg_gen_remu_i32(REG(a), REG(a), REG(b));

            }

            break;

        case 0x35: /* brk */

            {

                TCGv temp = tcg_temp_new_i32();

                tcg_gen_movi_i32(cpu_pc, ctx->pc);

                tcg_gen_movi_i32(temp, MOXIE_EX_BREAK);

                gen_helper_raise_exception(cpu_env, temp);

                tcg_temp_free_i32(temp);

            }

            break;

        case 0x36: /* ldo.b */

            {

                int a = (opcode >> 4) & 0xf;

                int b = opcode & 0xf;



                TCGv t1 = tcg_temp_new_i32();

                TCGv t2 = tcg_temp_new_i32();

                tcg_gen_addi_i32(t1, REG(b), cpu_ldl_code(env, ctx->pc+2));

                tcg_gen_qemu_ld8u(t2, t1, ctx->memidx);

                tcg_gen_mov_i32(REG(a), t2);



                tcg_temp_free_i32(t1);

                tcg_temp_free_i32(t2);



                length = 6;

            }

            break;

        case 0x37: /* sto.b */

            {

                int a = (opcode >> 4) & 0xf;

                int b = opcode & 0xf;



                TCGv t1 = tcg_temp_new_i32();

                TCGv t2 = tcg_temp_new_i32();

                tcg_gen_addi_i32(t1, REG(a), cpu_ldl_code(env, ctx->pc+2));

                tcg_gen_qemu_st8(REG(b), t1, ctx->memidx);



                tcg_temp_free_i32(t1);

                tcg_temp_free_i32(t2);



                length = 6;

            }

            break;

        case 0x38: /* ldo.s */

            {

                int a = (opcode >> 4) & 0xf;

                int b = opcode & 0xf;



                TCGv t1 = tcg_temp_new_i32();

                TCGv t2 = tcg_temp_new_i32();

                tcg_gen_addi_i32(t1, REG(b), cpu_ldl_code(env, ctx->pc+2));

                tcg_gen_qemu_ld16u(t2, t1, ctx->memidx);

                tcg_gen_mov_i32(REG(a), t2);



                tcg_temp_free_i32(t1);

                tcg_temp_free_i32(t2);



                length = 6;

            }

            break;

        case 0x39: /* sto.s */

            {

                int a = (opcode >> 4) & 0xf;

                int b = opcode & 0xf;



                TCGv t1 = tcg_temp_new_i32();

                TCGv t2 = tcg_temp_new_i32();

                tcg_gen_addi_i32(t1, REG(a), cpu_ldl_code(env, ctx->pc+2));

                tcg_gen_qemu_st16(REG(b), t1, ctx->memidx);

                tcg_temp_free_i32(t1);

                tcg_temp_free_i32(t2);



                length = 6;

            }

            break;

        default:

            {

                TCGv temp = tcg_temp_new_i32();

                tcg_gen_movi_i32(cpu_pc, ctx->pc);

                tcg_gen_movi_i32(temp, MOXIE_EX_BAD);

                gen_helper_raise_exception(cpu_env, temp);

                tcg_temp_free_i32(temp);

             }

            break;

        }

    }



    return length;

}
