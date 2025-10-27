static void tcg_constant_folding(TCGContext *s)

{

    int oi, oi_next, nb_temps, nb_globals;



    /* Array VALS has an element for each temp.

       If this temp holds a constant then its value is kept in VALS' element.

       If this temp is a copy of other ones then the other copies are

       available through the doubly linked circular list. */



    nb_temps = s->nb_temps;

    nb_globals = s->nb_globals;

    reset_all_temps(nb_temps);



    for (oi = s->gen_first_op_idx; oi >= 0; oi = oi_next) {

        tcg_target_ulong mask, partmask, affected;

        int nb_oargs, nb_iargs, i;

        TCGArg tmp;



        TCGOp * const op = &s->gen_op_buf[oi];

        TCGArg * const args = &s->gen_opparam_buf[op->args];

        TCGOpcode opc = op->opc;

        const TCGOpDef *def = &tcg_op_defs[opc];



        oi_next = op->next;

        if (opc == INDEX_op_call) {

            nb_oargs = op->callo;

            nb_iargs = op->calli;

        } else {

            nb_oargs = def->nb_oargs;

            nb_iargs = def->nb_iargs;

        }



        /* Do copy propagation */

        for (i = nb_oargs; i < nb_oargs + nb_iargs; i++) {

            if (temps[args[i]].state == TCG_TEMP_COPY) {

                args[i] = find_better_copy(s, args[i]);

            }

        }



        /* For commutative operations make constant second argument */

        switch (opc) {

        CASE_OP_32_64(add):

        CASE_OP_32_64(mul):

        CASE_OP_32_64(and):

        CASE_OP_32_64(or):

        CASE_OP_32_64(xor):

        CASE_OP_32_64(eqv):

        CASE_OP_32_64(nand):

        CASE_OP_32_64(nor):

        CASE_OP_32_64(muluh):

        CASE_OP_32_64(mulsh):

            swap_commutative(args[0], &args[1], &args[2]);

            break;

        CASE_OP_32_64(brcond):

            if (swap_commutative(-1, &args[0], &args[1])) {

                args[2] = tcg_swap_cond(args[2]);

            }

            break;

        CASE_OP_32_64(setcond):

            if (swap_commutative(args[0], &args[1], &args[2])) {

                args[3] = tcg_swap_cond(args[3]);

            }

            break;

        CASE_OP_32_64(movcond):

            if (swap_commutative(-1, &args[1], &args[2])) {

                args[5] = tcg_swap_cond(args[5]);

            }

            /* For movcond, we canonicalize the "false" input reg to match

               the destination reg so that the tcg backend can implement

               a "move if true" operation.  */

            if (swap_commutative(args[0], &args[4], &args[3])) {

                args[5] = tcg_invert_cond(args[5]);

            }

            break;

        CASE_OP_32_64(add2):

            swap_commutative(args[0], &args[2], &args[4]);

            swap_commutative(args[1], &args[3], &args[5]);

            break;

        CASE_OP_32_64(mulu2):

        CASE_OP_32_64(muls2):

            swap_commutative(args[0], &args[2], &args[3]);

            break;

        case INDEX_op_brcond2_i32:

            if (swap_commutative2(&args[0], &args[2])) {

                args[4] = tcg_swap_cond(args[4]);

            }

            break;

        case INDEX_op_setcond2_i32:

            if (swap_commutative2(&args[1], &args[3])) {

                args[5] = tcg_swap_cond(args[5]);

            }

            break;

        default:

            break;

        }



        /* Simplify expressions for "shift/rot r, 0, a => movi r, 0",

           and "sub r, 0, a => neg r, a" case.  */

        switch (opc) {

        CASE_OP_32_64(shl):

        CASE_OP_32_64(shr):

        CASE_OP_32_64(sar):

        CASE_OP_32_64(rotl):

        CASE_OP_32_64(rotr):

            if (temps[args[1]].state == TCG_TEMP_CONST

                && temps[args[1]].val == 0) {

                tcg_opt_gen_movi(s, op, args, args[0], 0);

                continue;

            }

            break;

        CASE_OP_32_64(sub):

            {

                TCGOpcode neg_op;

                bool have_neg;



                if (temps[args[2]].state == TCG_TEMP_CONST) {

                    /* Proceed with possible constant folding. */

                    break;

                }

                if (opc == INDEX_op_sub_i32) {

                    neg_op = INDEX_op_neg_i32;

                    have_neg = TCG_TARGET_HAS_neg_i32;

                } else {

                    neg_op = INDEX_op_neg_i64;

                    have_neg = TCG_TARGET_HAS_neg_i64;

                }

                if (!have_neg) {

                    break;

                }

                if (temps[args[1]].state == TCG_TEMP_CONST

                    && temps[args[1]].val == 0) {

                    op->opc = neg_op;

                    reset_temp(args[0]);

                    args[1] = args[2];

                    continue;

                }

            }

            break;

        CASE_OP_32_64(xor):

        CASE_OP_32_64(nand):

            if (temps[args[1]].state != TCG_TEMP_CONST

                && temps[args[2]].state == TCG_TEMP_CONST

                && temps[args[2]].val == -1) {

                i = 1;

                goto try_not;

            }

            break;

        CASE_OP_32_64(nor):

            if (temps[args[1]].state != TCG_TEMP_CONST

                && temps[args[2]].state == TCG_TEMP_CONST

                && temps[args[2]].val == 0) {

                i = 1;

                goto try_not;

            }

            break;

        CASE_OP_32_64(andc):

            if (temps[args[2]].state != TCG_TEMP_CONST

                && temps[args[1]].state == TCG_TEMP_CONST

                && temps[args[1]].val == -1) {

                i = 2;

                goto try_not;

            }

            break;

        CASE_OP_32_64(orc):

        CASE_OP_32_64(eqv):

            if (temps[args[2]].state != TCG_TEMP_CONST

                && temps[args[1]].state == TCG_TEMP_CONST

                && temps[args[1]].val == 0) {

                i = 2;

                goto try_not;

            }

            break;

        try_not:

            {

                TCGOpcode not_op;

                bool have_not;



                if (def->flags & TCG_OPF_64BIT) {

                    not_op = INDEX_op_not_i64;

                    have_not = TCG_TARGET_HAS_not_i64;

                } else {

                    not_op = INDEX_op_not_i32;

                    have_not = TCG_TARGET_HAS_not_i32;

                }

                if (!have_not) {

                    break;

                }

                op->opc = not_op;

                reset_temp(args[0]);

                args[1] = args[i];

                continue;

            }

        default:

            break;

        }



        /* Simplify expression for "op r, a, const => mov r, a" cases */

        switch (opc) {

        CASE_OP_32_64(add):

        CASE_OP_32_64(sub):

        CASE_OP_32_64(shl):

        CASE_OP_32_64(shr):

        CASE_OP_32_64(sar):

        CASE_OP_32_64(rotl):

        CASE_OP_32_64(rotr):

        CASE_OP_32_64(or):

        CASE_OP_32_64(xor):

        CASE_OP_32_64(andc):

            if (temps[args[1]].state != TCG_TEMP_CONST

                && temps[args[2]].state == TCG_TEMP_CONST

                && temps[args[2]].val == 0) {

                tcg_opt_gen_mov(s, op, args, args[0], args[1]);

                continue;

            }

            break;

        CASE_OP_32_64(and):

        CASE_OP_32_64(orc):

        CASE_OP_32_64(eqv):

            if (temps[args[1]].state != TCG_TEMP_CONST

                && temps[args[2]].state == TCG_TEMP_CONST

                && temps[args[2]].val == -1) {

                tcg_opt_gen_mov(s, op, args, args[0], args[1]);

                continue;

            }

            break;

        default:

            break;

        }



        /* Simplify using known-zero bits. Currently only ops with a single

           output argument is supported. */

        mask = -1;

        affected = -1;

        switch (opc) {

        CASE_OP_32_64(ext8s):

            if ((temps[args[1]].mask & 0x80) != 0) {

                break;

            }

        CASE_OP_32_64(ext8u):

            mask = 0xff;

            goto and_const;

        CASE_OP_32_64(ext16s):

            if ((temps[args[1]].mask & 0x8000) != 0) {

                break;

            }

        CASE_OP_32_64(ext16u):

            mask = 0xffff;

            goto and_const;

        case INDEX_op_ext32s_i64:

            if ((temps[args[1]].mask & 0x80000000) != 0) {

                break;

            }

        case INDEX_op_ext32u_i64:

            mask = 0xffffffffU;

            goto and_const;



        CASE_OP_32_64(and):

            mask = temps[args[2]].mask;

            if (temps[args[2]].state == TCG_TEMP_CONST) {

        and_const:

                affected = temps[args[1]].mask & ~mask;

            }

            mask = temps[args[1]].mask & mask;

            break;



        CASE_OP_32_64(andc):

            /* Known-zeros does not imply known-ones.  Therefore unless

               args[2] is constant, we can't infer anything from it.  */

            if (temps[args[2]].state == TCG_TEMP_CONST) {

                mask = ~temps[args[2]].mask;

                goto and_const;

            }

            /* But we certainly know nothing outside args[1] may be set. */

            mask = temps[args[1]].mask;

            break;



        case INDEX_op_sar_i32:

            if (temps[args[2]].state == TCG_TEMP_CONST) {

                tmp = temps[args[2]].val & 31;

                mask = (int32_t)temps[args[1]].mask >> tmp;

            }

            break;

        case INDEX_op_sar_i64:

            if (temps[args[2]].state == TCG_TEMP_CONST) {

                tmp = temps[args[2]].val & 63;

                mask = (int64_t)temps[args[1]].mask >> tmp;

            }

            break;



        case INDEX_op_shr_i32:

            if (temps[args[2]].state == TCG_TEMP_CONST) {

                tmp = temps[args[2]].val & 31;

                mask = (uint32_t)temps[args[1]].mask >> tmp;

            }

            break;

        case INDEX_op_shr_i64:

            if (temps[args[2]].state == TCG_TEMP_CONST) {

                tmp = temps[args[2]].val & 63;

                mask = (uint64_t)temps[args[1]].mask >> tmp;

            }

            break;



        case INDEX_op_trunc_shr_i32:

            mask = (uint64_t)temps[args[1]].mask >> args[2];

            break;



        CASE_OP_32_64(shl):

            if (temps[args[2]].state == TCG_TEMP_CONST) {

                tmp = temps[args[2]].val & (TCG_TARGET_REG_BITS - 1);

                mask = temps[args[1]].mask << tmp;

            }

            break;



        CASE_OP_32_64(neg):

            /* Set to 1 all bits to the left of the rightmost.  */

            mask = -(temps[args[1]].mask & -temps[args[1]].mask);

            break;



        CASE_OP_32_64(deposit):

            mask = deposit64(temps[args[1]].mask, args[3], args[4],

                             temps[args[2]].mask);

            break;



        CASE_OP_32_64(or):

        CASE_OP_32_64(xor):

            mask = temps[args[1]].mask | temps[args[2]].mask;

            break;



        CASE_OP_32_64(setcond):

        case INDEX_op_setcond2_i32:

            mask = 1;

            break;



        CASE_OP_32_64(movcond):

            mask = temps[args[3]].mask | temps[args[4]].mask;

            break;



        CASE_OP_32_64(ld8u):

            mask = 0xff;

            break;

        CASE_OP_32_64(ld16u):

            mask = 0xffff;

            break;

        case INDEX_op_ld32u_i64:

            mask = 0xffffffffu;

            break;



        CASE_OP_32_64(qemu_ld):

            {

                TCGMemOpIdx oi = args[nb_oargs + nb_iargs];

                TCGMemOp mop = get_memop(oi);

                if (!(mop & MO_SIGN)) {

                    mask = (2ULL << ((8 << (mop & MO_SIZE)) - 1)) - 1;

                }

            }

            break;



        default:

            break;

        }



        /* 32-bit ops generate 32-bit results.  For the result is zero test

           below, we can ignore high bits, but for further optimizations we

           need to record that the high bits contain garbage.  */

        partmask = mask;

        if (!(def->flags & TCG_OPF_64BIT)) {

            mask |= ~(tcg_target_ulong)0xffffffffu;

            partmask &= 0xffffffffu;

            affected &= 0xffffffffu;

        }



        if (partmask == 0) {

            assert(nb_oargs == 1);

            tcg_opt_gen_movi(s, op, args, args[0], 0);

            continue;

        }

        if (affected == 0) {

            assert(nb_oargs == 1);

            tcg_opt_gen_mov(s, op, args, args[0], args[1]);

            continue;

        }



        /* Simplify expression for "op r, a, 0 => movi r, 0" cases */

        switch (opc) {

        CASE_OP_32_64(and):

        CASE_OP_32_64(mul):

        CASE_OP_32_64(muluh):

        CASE_OP_32_64(mulsh):

            if ((temps[args[2]].state == TCG_TEMP_CONST

                && temps[args[2]].val == 0)) {

                tcg_opt_gen_movi(s, op, args, args[0], 0);

                continue;

            }

            break;

        default:

            break;

        }



        /* Simplify expression for "op r, a, a => mov r, a" cases */

        switch (opc) {

        CASE_OP_32_64(or):

        CASE_OP_32_64(and):

            if (temps_are_copies(args[1], args[2])) {

                tcg_opt_gen_mov(s, op, args, args[0], args[1]);

                continue;

            }

            break;

        default:

            break;

        }



        /* Simplify expression for "op r, a, a => movi r, 0" cases */

        switch (opc) {

        CASE_OP_32_64(andc):

        CASE_OP_32_64(sub):

        CASE_OP_32_64(xor):

            if (temps_are_copies(args[1], args[2])) {

                tcg_opt_gen_movi(s, op, args, args[0], 0);

                continue;

            }

            break;

        default:

            break;

        }



        /* Propagate constants through copy operations and do constant

           folding.  Constants will be substituted to arguments by register

           allocator where needed and possible.  Also detect copies. */

        switch (opc) {

        CASE_OP_32_64(mov):

            tcg_opt_gen_mov(s, op, args, args[0], args[1]);

            break;

        CASE_OP_32_64(movi):

            tcg_opt_gen_movi(s, op, args, args[0], args[1]);

            break;



        CASE_OP_32_64(not):

        CASE_OP_32_64(neg):

        CASE_OP_32_64(ext8s):

        CASE_OP_32_64(ext8u):

        CASE_OP_32_64(ext16s):

        CASE_OP_32_64(ext16u):

        case INDEX_op_ext32s_i64:

        case INDEX_op_ext32u_i64:

            if (temps[args[1]].state == TCG_TEMP_CONST) {

                tmp = do_constant_folding(opc, temps[args[1]].val, 0);

                tcg_opt_gen_movi(s, op, args, args[0], tmp);

                break;

            }

            goto do_default;



        case INDEX_op_trunc_shr_i32:

            if (temps[args[1]].state == TCG_TEMP_CONST) {

                tmp = do_constant_folding(opc, temps[args[1]].val, args[2]);

                tcg_opt_gen_movi(s, op, args, args[0], tmp);

                break;

            }

            goto do_default;



        CASE_OP_32_64(add):

        CASE_OP_32_64(sub):

        CASE_OP_32_64(mul):

        CASE_OP_32_64(or):

        CASE_OP_32_64(and):

        CASE_OP_32_64(xor):

        CASE_OP_32_64(shl):

        CASE_OP_32_64(shr):

        CASE_OP_32_64(sar):

        CASE_OP_32_64(rotl):

        CASE_OP_32_64(rotr):

        CASE_OP_32_64(andc):

        CASE_OP_32_64(orc):

        CASE_OP_32_64(eqv):

        CASE_OP_32_64(nand):

        CASE_OP_32_64(nor):

        CASE_OP_32_64(muluh):

        CASE_OP_32_64(mulsh):

        CASE_OP_32_64(div):

        CASE_OP_32_64(divu):

        CASE_OP_32_64(rem):

        CASE_OP_32_64(remu):

            if (temps[args[1]].state == TCG_TEMP_CONST

                && temps[args[2]].state == TCG_TEMP_CONST) {

                tmp = do_constant_folding(opc, temps[args[1]].val,

                                          temps[args[2]].val);

                tcg_opt_gen_movi(s, op, args, args[0], tmp);

                break;

            }

            goto do_default;



        CASE_OP_32_64(deposit):

            if (temps[args[1]].state == TCG_TEMP_CONST

                && temps[args[2]].state == TCG_TEMP_CONST) {

                tmp = deposit64(temps[args[1]].val, args[3], args[4],

                                temps[args[2]].val);

                tcg_opt_gen_movi(s, op, args, args[0], tmp);

                break;

            }

            goto do_default;



        CASE_OP_32_64(setcond):

            tmp = do_constant_folding_cond(opc, args[1], args[2], args[3]);

            if (tmp != 2) {

                tcg_opt_gen_movi(s, op, args, args[0], tmp);

                break;

            }

            goto do_default;



        CASE_OP_32_64(brcond):

            tmp = do_constant_folding_cond(opc, args[0], args[1], args[2]);

            if (tmp != 2) {

                if (tmp) {

                    reset_all_temps(nb_temps);

                    op->opc = INDEX_op_br;

                    args[0] = args[3];

                } else {

                    tcg_op_remove(s, op);

                }

                break;

            }

            goto do_default;



        CASE_OP_32_64(movcond):

            tmp = do_constant_folding_cond(opc, args[1], args[2], args[5]);

            if (tmp != 2) {

                tcg_opt_gen_mov(s, op, args, args[0], args[4-tmp]);

                break;

            }

            goto do_default;



        case INDEX_op_add2_i32:

        case INDEX_op_sub2_i32:

            if (temps[args[2]].state == TCG_TEMP_CONST

                && temps[args[3]].state == TCG_TEMP_CONST

                && temps[args[4]].state == TCG_TEMP_CONST

                && temps[args[5]].state == TCG_TEMP_CONST) {

                uint32_t al = temps[args[2]].val;

                uint32_t ah = temps[args[3]].val;

                uint32_t bl = temps[args[4]].val;

                uint32_t bh = temps[args[5]].val;

                uint64_t a = ((uint64_t)ah << 32) | al;

                uint64_t b = ((uint64_t)bh << 32) | bl;

                TCGArg rl, rh;

                TCGOp *op2 = insert_op_before(s, op, INDEX_op_movi_i32, 2);

                TCGArg *args2 = &s->gen_opparam_buf[op2->args];



                if (opc == INDEX_op_add2_i32) {

                    a += b;

                } else {

                    a -= b;

                }



                rl = args[0];

                rh = args[1];

                tcg_opt_gen_movi(s, op, args, rl, (uint32_t)a);

                tcg_opt_gen_movi(s, op2, args2, rh, (uint32_t)(a >> 32));



                /* We've done all we need to do with the movi.  Skip it.  */

                oi_next = op2->next;

                break;

            }

            goto do_default;



        case INDEX_op_mulu2_i32:

            if (temps[args[2]].state == TCG_TEMP_CONST

                && temps[args[3]].state == TCG_TEMP_CONST) {

                uint32_t a = temps[args[2]].val;

                uint32_t b = temps[args[3]].val;

                uint64_t r = (uint64_t)a * b;

                TCGArg rl, rh;

                TCGOp *op2 = insert_op_before(s, op, INDEX_op_movi_i32, 2);

                TCGArg *args2 = &s->gen_opparam_buf[op2->args];



                rl = args[0];

                rh = args[1];

                tcg_opt_gen_movi(s, op, args, rl, (uint32_t)r);

                tcg_opt_gen_movi(s, op2, args2, rh, (uint32_t)(r >> 32));



                /* We've done all we need to do with the movi.  Skip it.  */

                oi_next = op2->next;

                break;

            }

            goto do_default;



        case INDEX_op_brcond2_i32:

            tmp = do_constant_folding_cond2(&args[0], &args[2], args[4]);

            if (tmp != 2) {

                if (tmp) {

            do_brcond_true:

                    reset_all_temps(nb_temps);

                    op->opc = INDEX_op_br;

                    args[0] = args[5];

                } else {

            do_brcond_false:

                    tcg_op_remove(s, op);

                }

            } else if ((args[4] == TCG_COND_LT || args[4] == TCG_COND_GE)

                       && temps[args[2]].state == TCG_TEMP_CONST

                       && temps[args[3]].state == TCG_TEMP_CONST

                       && temps[args[2]].val == 0

                       && temps[args[3]].val == 0) {

                /* Simplify LT/GE comparisons vs zero to a single compare

                   vs the high word of the input.  */

            do_brcond_high:

                reset_all_temps(nb_temps);

                op->opc = INDEX_op_brcond_i32;

                args[0] = args[1];

                args[1] = args[3];

                args[2] = args[4];

                args[3] = args[5];

            } else if (args[4] == TCG_COND_EQ) {

                /* Simplify EQ comparisons where one of the pairs

                   can be simplified.  */

                tmp = do_constant_folding_cond(INDEX_op_brcond_i32,

                                               args[0], args[2], TCG_COND_EQ);

                if (tmp == 0) {

                    goto do_brcond_false;

                } else if (tmp == 1) {

                    goto do_brcond_high;

                }

                tmp = do_constant_folding_cond(INDEX_op_brcond_i32,

                                               args[1], args[3], TCG_COND_EQ);

                if (tmp == 0) {

                    goto do_brcond_false;

                } else if (tmp != 1) {

                    goto do_default;

                }

            do_brcond_low:

                reset_all_temps(nb_temps);

                op->opc = INDEX_op_brcond_i32;

                args[1] = args[2];

                args[2] = args[4];

                args[3] = args[5];

            } else if (args[4] == TCG_COND_NE) {

                /* Simplify NE comparisons where one of the pairs

                   can be simplified.  */

                tmp = do_constant_folding_cond(INDEX_op_brcond_i32,

                                               args[0], args[2], TCG_COND_NE);

                if (tmp == 0) {

                    goto do_brcond_high;

                } else if (tmp == 1) {

                    goto do_brcond_true;

                }

                tmp = do_constant_folding_cond(INDEX_op_brcond_i32,

                                               args[1], args[3], TCG_COND_NE);

                if (tmp == 0) {

                    goto do_brcond_low;

                } else if (tmp == 1) {

                    goto do_brcond_true;

                }

                goto do_default;

            } else {

                goto do_default;

            }

            break;



        case INDEX_op_setcond2_i32:

            tmp = do_constant_folding_cond2(&args[1], &args[3], args[5]);

            if (tmp != 2) {

            do_setcond_const:

                tcg_opt_gen_movi(s, op, args, args[0], tmp);

            } else if ((args[5] == TCG_COND_LT || args[5] == TCG_COND_GE)

                       && temps[args[3]].state == TCG_TEMP_CONST

                       && temps[args[4]].state == TCG_TEMP_CONST

                       && temps[args[3]].val == 0

                       && temps[args[4]].val == 0) {

                /* Simplify LT/GE comparisons vs zero to a single compare

                   vs the high word of the input.  */

            do_setcond_high:

                reset_temp(args[0]);

                temps[args[0]].mask = 1;

                op->opc = INDEX_op_setcond_i32;

                args[1] = args[2];

                args[2] = args[4];

                args[3] = args[5];

            } else if (args[5] == TCG_COND_EQ) {

                /* Simplify EQ comparisons where one of the pairs

                   can be simplified.  */

                tmp = do_constant_folding_cond(INDEX_op_setcond_i32,

                                               args[1], args[3], TCG_COND_EQ);

                if (tmp == 0) {

                    goto do_setcond_const;

                } else if (tmp == 1) {

                    goto do_setcond_high;

                }

                tmp = do_constant_folding_cond(INDEX_op_setcond_i32,

                                               args[2], args[4], TCG_COND_EQ);

                if (tmp == 0) {

                    goto do_setcond_high;

                } else if (tmp != 1) {

                    goto do_default;

                }

            do_setcond_low:

                reset_temp(args[0]);

                temps[args[0]].mask = 1;

                op->opc = INDEX_op_setcond_i32;

                args[2] = args[3];

                args[3] = args[5];

            } else if (args[5] == TCG_COND_NE) {

                /* Simplify NE comparisons where one of the pairs

                   can be simplified.  */

                tmp = do_constant_folding_cond(INDEX_op_setcond_i32,

                                               args[1], args[3], TCG_COND_NE);

                if (tmp == 0) {

                    goto do_setcond_high;

                } else if (tmp == 1) {

                    goto do_setcond_const;

                }

                tmp = do_constant_folding_cond(INDEX_op_setcond_i32,

                                               args[2], args[4], TCG_COND_NE);

                if (tmp == 0) {

                    goto do_setcond_low;

                } else if (tmp == 1) {

                    goto do_setcond_const;

                }

                goto do_default;

            } else {

                goto do_default;

            }

            break;



        case INDEX_op_call:

            if (!(args[nb_oargs + nb_iargs + 1]

                  & (TCG_CALL_NO_READ_GLOBALS | TCG_CALL_NO_WRITE_GLOBALS))) {

                for (i = 0; i < nb_globals; i++) {

                    reset_temp(i);

                }

            }

            goto do_reset_output;



        default:

        do_default:

            /* Default case: we know nothing about operation (or were unable

               to compute the operation result) so no propagation is done.

               We trash everything if the operation is the end of a basic

               block, otherwise we only trash the output args.  "mask" is

               the non-zero bits mask for the first output arg.  */

            if (def->flags & TCG_OPF_BB_END) {

                reset_all_temps(nb_temps);

            } else {

        do_reset_output:

                for (i = 0; i < nb_oargs; i++) {

                    reset_temp(args[i]);

                    /* Save the corresponding known-zero bits mask for the

                       first output argument (only one supported so far). */

                    if (i == 0) {

                        temps[args[i]].mask = mask;

                    }

                }

            }

            break;

        }

    }

}
