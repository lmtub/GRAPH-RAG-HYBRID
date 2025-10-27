void tcg_optimize(TCGContext *s)

{

    int oi, oi_next, nb_temps, nb_globals;

    TCGOp *prev_mb = NULL;

    struct tcg_temp_info *infos;

    TCGTempSet temps_used;



    /* Array VALS has an element for each temp.

       If this temp holds a constant then its value is kept in VALS' element.

       If this temp is a copy of other ones then the other copies are

       available through the doubly linked circular list. */



    nb_temps = s->nb_temps;

    nb_globals = s->nb_globals;

    bitmap_zero(temps_used.l, nb_temps);

    infos = tcg_malloc(sizeof(struct tcg_temp_info) * nb_temps);



    for (oi = s->gen_op_buf[0].next; oi != 0; oi = oi_next) {

        tcg_target_ulong mask, partmask, affected;

        int nb_oargs, nb_iargs, i;

        TCGArg tmp;



        TCGOp * const op = &s->gen_op_buf[oi];

        TCGOpcode opc = op->opc;

        const TCGOpDef *def = &tcg_op_defs[opc];



        oi_next = op->next;



        /* Count the arguments, and initialize the temps that are

           going to be used */

        if (opc == INDEX_op_call) {

            nb_oargs = op->callo;

            nb_iargs = op->calli;

            for (i = 0; i < nb_oargs + nb_iargs; i++) {

                TCGTemp *ts = arg_temp(op->args[i]);

                if (ts) {

                    init_ts_info(infos, &temps_used, ts);

                }

            }

        } else {

            nb_oargs = def->nb_oargs;

            nb_iargs = def->nb_iargs;

            for (i = 0; i < nb_oargs + nb_iargs; i++) {

                init_arg_info(infos, &temps_used, op->args[i]);

            }

        }



        /* Do copy propagation */

        for (i = nb_oargs; i < nb_oargs + nb_iargs; i++) {

            TCGTemp *ts = arg_temp(op->args[i]);

            if (ts && ts_is_copy(ts)) {

                op->args[i] = temp_arg(find_better_copy(s, ts));

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

            swap_commutative(op->args[0], &op->args[1], &op->args[2]);

            break;

        CASE_OP_32_64(brcond):

            if (swap_commutative(-1, &op->args[0], &op->args[1])) {

                op->args[2] = tcg_swap_cond(op->args[2]);

            }

            break;

        CASE_OP_32_64(setcond):

            if (swap_commutative(op->args[0], &op->args[1], &op->args[2])) {

                op->args[3] = tcg_swap_cond(op->args[3]);

            }

            break;

        CASE_OP_32_64(movcond):

            if (swap_commutative(-1, &op->args[1], &op->args[2])) {

                op->args[5] = tcg_swap_cond(op->args[5]);

            }

            /* For movcond, we canonicalize the "false" input reg to match

               the destination reg so that the tcg backend can implement

               a "move if true" operation.  */

            if (swap_commutative(op->args[0], &op->args[4], &op->args[3])) {

                op->args[5] = tcg_invert_cond(op->args[5]);

            }

            break;

        CASE_OP_32_64(add2):

            swap_commutative(op->args[0], &op->args[2], &op->args[4]);

            swap_commutative(op->args[1], &op->args[3], &op->args[5]);

            break;

        CASE_OP_32_64(mulu2):

        CASE_OP_32_64(muls2):

            swap_commutative(op->args[0], &op->args[2], &op->args[3]);

            break;

        case INDEX_op_brcond2_i32:

            if (swap_commutative2(&op->args[0], &op->args[2])) {

                op->args[4] = tcg_swap_cond(op->args[4]);

            }

            break;

        case INDEX_op_setcond2_i32:

            if (swap_commutative2(&op->args[1], &op->args[3])) {

                op->args[5] = tcg_swap_cond(op->args[5]);

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

            if (arg_is_const(op->args[1])

                && arg_info(op->args[1])->val == 0) {

                tcg_opt_gen_movi(s, op, op->args[0], 0);

                continue;

            }

            break;

        CASE_OP_32_64(sub):

            {

                TCGOpcode neg_op;

                bool have_neg;



                if (arg_is_const(op->args[2])) {

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

                if (arg_is_const(op->args[1])

                    && arg_info(op->args[1])->val == 0) {

                    op->opc = neg_op;

                    reset_temp(op->args[0]);

                    op->args[1] = op->args[2];

                    continue;

                }

            }

            break;

        CASE_OP_32_64(xor):

        CASE_OP_32_64(nand):

            if (!arg_is_const(op->args[1])

                && arg_is_const(op->args[2])

                && arg_info(op->args[2])->val == -1) {

                i = 1;

                goto try_not;

            }

            break;

        CASE_OP_32_64(nor):

            if (!arg_is_const(op->args[1])

                && arg_is_const(op->args[2])

                && arg_info(op->args[2])->val == 0) {

                i = 1;

                goto try_not;

            }

            break;

        CASE_OP_32_64(andc):

            if (!arg_is_const(op->args[2])

                && arg_is_const(op->args[1])

                && arg_info(op->args[1])->val == -1) {

                i = 2;

                goto try_not;

            }

            break;

        CASE_OP_32_64(orc):

        CASE_OP_32_64(eqv):

            if (!arg_is_const(op->args[2])

                && arg_is_const(op->args[1])

                && arg_info(op->args[1])->val == 0) {

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

                reset_temp(op->args[0]);

                op->args[1] = op->args[i];

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

            if (!arg_is_const(op->args[1])

                && arg_is_const(op->args[2])

                && arg_info(op->args[2])->val == 0) {

                tcg_opt_gen_mov(s, op, op->args[0], op->args[1]);

                continue;

            }

            break;

        CASE_OP_32_64(and):

        CASE_OP_32_64(orc):

        CASE_OP_32_64(eqv):

            if (!arg_is_const(op->args[1])

                && arg_is_const(op->args[2])

                && arg_info(op->args[2])->val == -1) {

                tcg_opt_gen_mov(s, op, op->args[0], op->args[1]);

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

            if ((arg_info(op->args[1])->mask & 0x80) != 0) {

                break;

            }

        CASE_OP_32_64(ext8u):

            mask = 0xff;

            goto and_const;

        CASE_OP_32_64(ext16s):

            if ((arg_info(op->args[1])->mask & 0x8000) != 0) {

                break;

            }

        CASE_OP_32_64(ext16u):

            mask = 0xffff;

            goto and_const;

        case INDEX_op_ext32s_i64:

            if ((arg_info(op->args[1])->mask & 0x80000000) != 0) {

                break;

            }

        case INDEX_op_ext32u_i64:

            mask = 0xffffffffU;

            goto and_const;



        CASE_OP_32_64(and):

            mask = arg_info(op->args[2])->mask;

            if (arg_is_const(op->args[2])) {

        and_const:

                affected = arg_info(op->args[1])->mask & ~mask;

            }

            mask = arg_info(op->args[1])->mask & mask;

            break;



        case INDEX_op_ext_i32_i64:

            if ((arg_info(op->args[1])->mask & 0x80000000) != 0) {

                break;

            }

        case INDEX_op_extu_i32_i64:

            /* We do not compute affected as it is a size changing op.  */

            mask = (uint32_t)arg_info(op->args[1])->mask;

            break;



        CASE_OP_32_64(andc):

            /* Known-zeros does not imply known-ones.  Therefore unless

               op->args[2] is constant, we can't infer anything from it.  */

            if (arg_is_const(op->args[2])) {

                mask = ~arg_info(op->args[2])->mask;

                goto and_const;

            }

            /* But we certainly know nothing outside args[1] may be set. */

            mask = arg_info(op->args[1])->mask;

            break;



        case INDEX_op_sar_i32:

            if (arg_is_const(op->args[2])) {

                tmp = arg_info(op->args[2])->val & 31;

                mask = (int32_t)arg_info(op->args[1])->mask >> tmp;

            }

            break;

        case INDEX_op_sar_i64:

            if (arg_is_const(op->args[2])) {

                tmp = arg_info(op->args[2])->val & 63;

                mask = (int64_t)arg_info(op->args[1])->mask >> tmp;

            }

            break;



        case INDEX_op_shr_i32:

            if (arg_is_const(op->args[2])) {

                tmp = arg_info(op->args[2])->val & 31;

                mask = (uint32_t)arg_info(op->args[1])->mask >> tmp;

            }

            break;

        case INDEX_op_shr_i64:

            if (arg_is_const(op->args[2])) {

                tmp = arg_info(op->args[2])->val & 63;

                mask = (uint64_t)arg_info(op->args[1])->mask >> tmp;

            }

            break;



        case INDEX_op_extrl_i64_i32:

            mask = (uint32_t)arg_info(op->args[1])->mask;

            break;

        case INDEX_op_extrh_i64_i32:

            mask = (uint64_t)arg_info(op->args[1])->mask >> 32;

            break;



        CASE_OP_32_64(shl):

            if (arg_is_const(op->args[2])) {

                tmp = arg_info(op->args[2])->val & (TCG_TARGET_REG_BITS - 1);

                mask = arg_info(op->args[1])->mask << tmp;

            }

            break;



        CASE_OP_32_64(neg):

            /* Set to 1 all bits to the left of the rightmost.  */

            mask = -(arg_info(op->args[1])->mask

                     & -arg_info(op->args[1])->mask);

            break;



        CASE_OP_32_64(deposit):

            mask = deposit64(arg_info(op->args[1])->mask,

                             op->args[3], op->args[4],

                             arg_info(op->args[2])->mask);

            break;



        CASE_OP_32_64(extract):

            mask = extract64(arg_info(op->args[1])->mask,

                             op->args[2], op->args[3]);

            if (op->args[2] == 0) {

                affected = arg_info(op->args[1])->mask & ~mask;

            }

            break;

        CASE_OP_32_64(sextract):

            mask = sextract64(arg_info(op->args[1])->mask,

                              op->args[2], op->args[3]);

            if (op->args[2] == 0 && (tcg_target_long)mask >= 0) {

                affected = arg_info(op->args[1])->mask & ~mask;

            }

            break;



        CASE_OP_32_64(or):

        CASE_OP_32_64(xor):

            mask = arg_info(op->args[1])->mask | arg_info(op->args[2])->mask;

            break;



        case INDEX_op_clz_i32:

        case INDEX_op_ctz_i32:

            mask = arg_info(op->args[2])->mask | 31;

            break;



        case INDEX_op_clz_i64:

        case INDEX_op_ctz_i64:

            mask = arg_info(op->args[2])->mask | 63;

            break;



        case INDEX_op_ctpop_i32:

            mask = 32 | 31;

            break;

        case INDEX_op_ctpop_i64:

            mask = 64 | 63;

            break;



        CASE_OP_32_64(setcond):

        case INDEX_op_setcond2_i32:

            mask = 1;

            break;



        CASE_OP_32_64(movcond):

            mask = arg_info(op->args[3])->mask | arg_info(op->args[4])->mask;

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

                TCGMemOpIdx oi = op->args[nb_oargs + nb_iargs];

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

            tcg_debug_assert(nb_oargs == 1);

            tcg_opt_gen_movi(s, op, op->args[0], 0);

            continue;

        }

        if (affected == 0) {

            tcg_debug_assert(nb_oargs == 1);

            tcg_opt_gen_mov(s, op, op->args[0], op->args[1]);

            continue;

        }



        /* Simplify expression for "op r, a, 0 => movi r, 0" cases */

        switch (opc) {

        CASE_OP_32_64(and):

        CASE_OP_32_64(mul):

        CASE_OP_32_64(muluh):

        CASE_OP_32_64(mulsh):

            if (arg_is_const(op->args[2])

                && arg_info(op->args[2])->val == 0) {

                tcg_opt_gen_movi(s, op, op->args[0], 0);

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

            if (args_are_copies(op->args[1], op->args[2])) {

                tcg_opt_gen_mov(s, op, op->args[0], op->args[1]);

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

            if (args_are_copies(op->args[1], op->args[2])) {

                tcg_opt_gen_movi(s, op, op->args[0], 0);

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

            tcg_opt_gen_mov(s, op, op->args[0], op->args[1]);

            break;

        CASE_OP_32_64(movi):

            tcg_opt_gen_movi(s, op, op->args[0], op->args[1]);

            break;



        CASE_OP_32_64(not):

        CASE_OP_32_64(neg):

        CASE_OP_32_64(ext8s):

        CASE_OP_32_64(ext8u):

        CASE_OP_32_64(ext16s):

        CASE_OP_32_64(ext16u):

        CASE_OP_32_64(ctpop):

        case INDEX_op_ext32s_i64:

        case INDEX_op_ext32u_i64:

        case INDEX_op_ext_i32_i64:

        case INDEX_op_extu_i32_i64:

        case INDEX_op_extrl_i64_i32:

        case INDEX_op_extrh_i64_i32:

            if (arg_is_const(op->args[1])) {

                tmp = do_constant_folding(opc, arg_info(op->args[1])->val, 0);

                tcg_opt_gen_movi(s, op, op->args[0], tmp);

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

            if (arg_is_const(op->args[1]) && arg_is_const(op->args[2])) {

                tmp = do_constant_folding(opc, arg_info(op->args[1])->val,

                                          arg_info(op->args[2])->val);

                tcg_opt_gen_movi(s, op, op->args[0], tmp);

                break;

            }

            goto do_default;



        CASE_OP_32_64(clz):

        CASE_OP_32_64(ctz):

            if (arg_is_const(op->args[1])) {

                TCGArg v = arg_info(op->args[1])->val;

                if (v != 0) {

                    tmp = do_constant_folding(opc, v, 0);

                    tcg_opt_gen_movi(s, op, op->args[0], tmp);

                } else {

                    tcg_opt_gen_mov(s, op, op->args[0], op->args[2]);

                }

                break;

            }

            goto do_default;



        CASE_OP_32_64(deposit):

            if (arg_is_const(op->args[1]) && arg_is_const(op->args[2])) {

                tmp = deposit64(arg_info(op->args[1])->val,

                                op->args[3], op->args[4],

                                arg_info(op->args[2])->val);

                tcg_opt_gen_movi(s, op, op->args[0], tmp);

                break;

            }

            goto do_default;



        CASE_OP_32_64(extract):

            if (arg_is_const(op->args[1])) {

                tmp = extract64(arg_info(op->args[1])->val,

                                op->args[2], op->args[3]);

                tcg_opt_gen_movi(s, op, op->args[0], tmp);

                break;

            }

            goto do_default;



        CASE_OP_32_64(sextract):

            if (arg_is_const(op->args[1])) {

                tmp = sextract64(arg_info(op->args[1])->val,

                                 op->args[2], op->args[3]);

                tcg_opt_gen_movi(s, op, op->args[0], tmp);

                break;

            }

            goto do_default;



        CASE_OP_32_64(setcond):

            tmp = do_constant_folding_cond(opc, op->args[1],

                                           op->args[2], op->args[3]);

            if (tmp != 2) {

                tcg_opt_gen_movi(s, op, op->args[0], tmp);

                break;

            }

            goto do_default;



        CASE_OP_32_64(brcond):

            tmp = do_constant_folding_cond(opc, op->args[0],

                                           op->args[1], op->args[2]);

            if (tmp != 2) {

                if (tmp) {

                    bitmap_zero(temps_used.l, nb_temps);

                    op->opc = INDEX_op_br;

                    op->args[0] = op->args[3];

                } else {

                    tcg_op_remove(s, op);

                }

                break;

            }

            goto do_default;



        CASE_OP_32_64(movcond):

            tmp = do_constant_folding_cond(opc, op->args[1],

                                           op->args[2], op->args[5]);

            if (tmp != 2) {

                tcg_opt_gen_mov(s, op, op->args[0], op->args[4-tmp]);

                break;

            }

            if (arg_is_const(op->args[3]) && arg_is_const(op->args[4])) {

                tcg_target_ulong tv = arg_info(op->args[3])->val;

                tcg_target_ulong fv = arg_info(op->args[4])->val;

                TCGCond cond = op->args[5];

                if (fv == 1 && tv == 0) {

                    cond = tcg_invert_cond(cond);

                } else if (!(tv == 1 && fv == 0)) {

                    goto do_default;

                }

                op->args[3] = cond;

                op->opc = opc = (opc == INDEX_op_movcond_i32

                                 ? INDEX_op_setcond_i32

                                 : INDEX_op_setcond_i64);

                nb_iargs = 2;

            }

            goto do_default;



        case INDEX_op_add2_i32:

        case INDEX_op_sub2_i32:

            if (arg_is_const(op->args[2]) && arg_is_const(op->args[3])

                && arg_is_const(op->args[4]) && arg_is_const(op->args[5])) {

                uint32_t al = arg_info(op->args[2])->val;

                uint32_t ah = arg_info(op->args[3])->val;

                uint32_t bl = arg_info(op->args[4])->val;

                uint32_t bh = arg_info(op->args[5])->val;

                uint64_t a = ((uint64_t)ah << 32) | al;

                uint64_t b = ((uint64_t)bh << 32) | bl;

                TCGArg rl, rh;

                TCGOp *op2 = tcg_op_insert_before(s, op, INDEX_op_movi_i32, 2);



                if (opc == INDEX_op_add2_i32) {

                    a += b;

                } else {

                    a -= b;

                }



                rl = op->args[0];

                rh = op->args[1];

                tcg_opt_gen_movi(s, op, rl, (int32_t)a);

                tcg_opt_gen_movi(s, op2, rh, (int32_t)(a >> 32));



                /* We've done all we need to do with the movi.  Skip it.  */

                oi_next = op2->next;

                break;

            }

            goto do_default;



        case INDEX_op_mulu2_i32:

            if (arg_is_const(op->args[2]) && arg_is_const(op->args[3])) {

                uint32_t a = arg_info(op->args[2])->val;

                uint32_t b = arg_info(op->args[3])->val;

                uint64_t r = (uint64_t)a * b;

                TCGArg rl, rh;

                TCGOp *op2 = tcg_op_insert_before(s, op, INDEX_op_movi_i32, 2);



                rl = op->args[0];

                rh = op->args[1];

                tcg_opt_gen_movi(s, op, rl, (int32_t)r);

                tcg_opt_gen_movi(s, op2, rh, (int32_t)(r >> 32));



                /* We've done all we need to do with the movi.  Skip it.  */

                oi_next = op2->next;

                break;

            }

            goto do_default;



        case INDEX_op_brcond2_i32:

            tmp = do_constant_folding_cond2(&op->args[0], &op->args[2],

                                            op->args[4]);

            if (tmp != 2) {

                if (tmp) {

            do_brcond_true:

                    bitmap_zero(temps_used.l, nb_temps);

                    op->opc = INDEX_op_br;

                    op->args[0] = op->args[5];

                } else {

            do_brcond_false:

                    tcg_op_remove(s, op);

                }

            } else if ((op->args[4] == TCG_COND_LT

                        || op->args[4] == TCG_COND_GE)

                       && arg_is_const(op->args[2])

                       && arg_info(op->args[2])->val == 0

                       && arg_is_const(op->args[3])

                       && arg_info(op->args[3])->val == 0) {

                /* Simplify LT/GE comparisons vs zero to a single compare

                   vs the high word of the input.  */

            do_brcond_high:

                bitmap_zero(temps_used.l, nb_temps);

                op->opc = INDEX_op_brcond_i32;

                op->args[0] = op->args[1];

                op->args[1] = op->args[3];

                op->args[2] = op->args[4];

                op->args[3] = op->args[5];

            } else if (op->args[4] == TCG_COND_EQ) {

                /* Simplify EQ comparisons where one of the pairs

                   can be simplified.  */

                tmp = do_constant_folding_cond(INDEX_op_brcond_i32,

                                               op->args[0], op->args[2],

                                               TCG_COND_EQ);

                if (tmp == 0) {

                    goto do_brcond_false;

                } else if (tmp == 1) {

                    goto do_brcond_high;

                }

                tmp = do_constant_folding_cond(INDEX_op_brcond_i32,

                                               op->args[1], op->args[3],

                                               TCG_COND_EQ);

                if (tmp == 0) {

                    goto do_brcond_false;

                } else if (tmp != 1) {

                    goto do_default;

                }

            do_brcond_low:

                bitmap_zero(temps_used.l, nb_temps);

                op->opc = INDEX_op_brcond_i32;

                op->args[1] = op->args[2];

                op->args[2] = op->args[4];

                op->args[3] = op->args[5];

            } else if (op->args[4] == TCG_COND_NE) {

                /* Simplify NE comparisons where one of the pairs

                   can be simplified.  */

                tmp = do_constant_folding_cond(INDEX_op_brcond_i32,

                                               op->args[0], op->args[2],

                                               TCG_COND_NE);

                if (tmp == 0) {

                    goto do_brcond_high;

                } else if (tmp == 1) {

                    goto do_brcond_true;

                }

                tmp = do_constant_folding_cond(INDEX_op_brcond_i32,

                                               op->args[1], op->args[3],

                                               TCG_COND_NE);

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

            tmp = do_constant_folding_cond2(&op->args[1], &op->args[3],

                                            op->args[5]);

            if (tmp != 2) {

            do_setcond_const:

                tcg_opt_gen_movi(s, op, op->args[0], tmp);

            } else if ((op->args[5] == TCG_COND_LT

                        || op->args[5] == TCG_COND_GE)

                       && arg_is_const(op->args[3])

                       && arg_info(op->args[3])->val == 0

                       && arg_is_const(op->args[4])

                       && arg_info(op->args[4])->val == 0) {

                /* Simplify LT/GE comparisons vs zero to a single compare

                   vs the high word of the input.  */

            do_setcond_high:

                reset_temp(op->args[0]);

                arg_info(op->args[0])->mask = 1;

                op->opc = INDEX_op_setcond_i32;

                op->args[1] = op->args[2];

                op->args[2] = op->args[4];

                op->args[3] = op->args[5];

            } else if (op->args[5] == TCG_COND_EQ) {

                /* Simplify EQ comparisons where one of the pairs

                   can be simplified.  */

                tmp = do_constant_folding_cond(INDEX_op_setcond_i32,

                                               op->args[1], op->args[3],

                                               TCG_COND_EQ);

                if (tmp == 0) {

                    goto do_setcond_const;

                } else if (tmp == 1) {

                    goto do_setcond_high;

                }

                tmp = do_constant_folding_cond(INDEX_op_setcond_i32,

                                               op->args[2], op->args[4],

                                               TCG_COND_EQ);

                if (tmp == 0) {

                    goto do_setcond_high;

                } else if (tmp != 1) {

                    goto do_default;

                }

            do_setcond_low:

                reset_temp(op->args[0]);

                arg_info(op->args[0])->mask = 1;

                op->opc = INDEX_op_setcond_i32;

                op->args[2] = op->args[3];

                op->args[3] = op->args[5];

            } else if (op->args[5] == TCG_COND_NE) {

                /* Simplify NE comparisons where one of the pairs

                   can be simplified.  */

                tmp = do_constant_folding_cond(INDEX_op_setcond_i32,

                                               op->args[1], op->args[3],

                                               TCG_COND_NE);

                if (tmp == 0) {

                    goto do_setcond_high;

                } else if (tmp == 1) {

                    goto do_setcond_const;

                }

                tmp = do_constant_folding_cond(INDEX_op_setcond_i32,

                                               op->args[2], op->args[4],

                                               TCG_COND_NE);

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

            if (!(op->args[nb_oargs + nb_iargs + 1]

                  & (TCG_CALL_NO_READ_GLOBALS | TCG_CALL_NO_WRITE_GLOBALS))) {

                for (i = 0; i < nb_globals; i++) {

                    if (test_bit(i, temps_used.l)) {

                        reset_ts(&s->temps[i]);

                    }

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

                bitmap_zero(temps_used.l, nb_temps);

            } else {

        do_reset_output:

                for (i = 0; i < nb_oargs; i++) {

                    reset_temp(op->args[i]);

                    /* Save the corresponding known-zero bits mask for the

                       first output argument (only one supported so far). */

                    if (i == 0) {

                        arg_info(op->args[i])->mask = mask;

                    }

                }

            }

            break;

        }



        /* Eliminate duplicate and redundant fence instructions.  */

        if (prev_mb) {

            switch (opc) {

            case INDEX_op_mb:

                /* Merge two barriers of the same type into one,

                 * or a weaker barrier into a stronger one,

                 * or two weaker barriers into a stronger one.

                 *   mb X; mb Y => mb X|Y

                 *   mb; strl => mb; st

                 *   ldaq; mb => ld; mb

                 *   ldaq; strl => ld; mb; st

                 * Other combinations are also merged into a strong

                 * barrier.  This is stricter than specified but for

                 * the purposes of TCG is better than not optimizing.

                 */

                prev_mb->args[0] |= op->args[0];

                tcg_op_remove(s, op);

                break;



            default:

                /* Opcodes that end the block stop the optimization.  */

                if ((def->flags & TCG_OPF_BB_END) == 0) {

                    break;

                }

                /* fallthru */

            case INDEX_op_qemu_ld_i32:

            case INDEX_op_qemu_ld_i64:

            case INDEX_op_qemu_st_i32:

            case INDEX_op_qemu_st_i64:

            case INDEX_op_call:

                /* Opcodes that touch guest memory stop the optimization.  */

                prev_mb = NULL;

                break;

            }

        } else if (opc == INDEX_op_mb) {

            prev_mb = op;

        }

    }

}
