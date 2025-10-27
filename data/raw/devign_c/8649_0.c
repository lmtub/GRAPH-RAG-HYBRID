static int tcg_match_add2i(TCGType type, tcg_target_long val)

{

    if (facilities & FACILITY_EXT_IMM) {

        if (type == TCG_TYPE_I32) {

            return 1;

        } else if (val >= -0xffffffffll && val <= 0xffffffffll) {

            return 1;

        }

    }

    return 0;

}
