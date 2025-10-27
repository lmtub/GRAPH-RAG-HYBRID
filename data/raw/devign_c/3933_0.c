static void tcg_out_dat_rIK(TCGContext *s, int cond, int opc, int opinv,

                            TCGReg dst, TCGReg lhs, TCGArg rhs,

                            bool rhs_is_const)

{

    /* Emit either the reg,imm or reg,reg form of a data-processing insn.

     * rhs must satisfy the "rIK" constraint.

     */

    if (rhs_is_const) {

        int rot = encode_imm(rhs);

        if (rot < 0) {

            rhs = ~rhs;

            rot = encode_imm(rhs);

            assert(rot >= 0);

            opc = opinv;

        }

        tcg_out_dat_imm(s, cond, opc, dst, lhs, rotl(rhs, rot) | (rot << 7));

    } else {

        tcg_out_dat_reg(s, cond, opc, dst, lhs, rhs, SHIFT_IMM_LSL(0));

    }

}
