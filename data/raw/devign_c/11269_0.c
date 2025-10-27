static void tcg_opt_gen_movi(TCGContext *s, TCGOp *op, TCGArg *args,

                             TCGArg dst, TCGArg val)

{

    TCGOpcode new_op = op_to_movi(op->opc);

    tcg_target_ulong mask;



    op->opc = new_op;



    reset_temp(dst);

    temps[dst].state = TCG_TEMP_CONST;

    temps[dst].val = val;

    mask = val;

    if (TCG_TARGET_REG_BITS > 32 && new_op == INDEX_op_mov_i32) {

        /* High bits of the destination are now garbage.  */

        mask |= ~0xffffffffull;

    }

    temps[dst].mask = mask;



    args[0] = dst;

    args[1] = val;

}
