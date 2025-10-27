static void tcg_out_st(TCGContext *s, TCGType type, TCGReg arg, TCGReg arg1,

                       intptr_t arg2)

{

    uint8_t *old_code_ptr = s->code_ptr;

    if (type == TCG_TYPE_I32) {

        tcg_out_op_t(s, INDEX_op_st_i32);

        tcg_out_r(s, arg);

        tcg_out_r(s, arg1);

        tcg_out32(s, arg2);

    } else {

        assert(type == TCG_TYPE_I64);

#if TCG_TARGET_REG_BITS == 64

        tcg_out_op_t(s, INDEX_op_st_i64);

        tcg_out_r(s, arg);

        tcg_out_r(s, arg1);

        tcg_out32(s, arg2);

#else

        TODO();

#endif

    }

    old_code_ptr[1] = s->code_ptr - old_code_ptr;

}
