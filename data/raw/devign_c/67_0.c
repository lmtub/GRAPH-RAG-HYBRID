static int tcg_match_cmpi(TCGType type, tcg_target_long val)

{

    if (facilities & FACILITY_EXT_IMM) {

        /* The COMPARE IMMEDIATE instruction is available.  */

        if (type == TCG_TYPE_I32) {

            /* We have a 32-bit immediate and can compare against anything.  */

            return 1;

        } else {

            /* ??? We have no insight here into whether the comparison is

               signed or unsigned.  The COMPARE IMMEDIATE insn uses a 32-bit

               signed immediate, and the COMPARE LOGICAL IMMEDIATE insn uses

               a 32-bit unsigned immediate.  If we were to use the (semi)

               obvious "val == (int32_t)val" we would be enabling unsigned

               comparisons vs very large numbers.  The only solution is to

               take the intersection of the ranges.  */

            /* ??? Another possible solution is to simply lie and allow all

               constants here and force the out-of-range values into a temp

               register in tgen_cmp when we have knowledge of the actual

               comparison code in use.  */

            return val >= 0 && val <= 0x7fffffff;

        }

    } else {

        /* Only the LOAD AND TEST instruction is available.  */

        return val == 0;

    }

}
