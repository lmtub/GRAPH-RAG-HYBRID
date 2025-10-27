static inline TCGv gen_extend(TCGv val, int opsize, int sign)

{

    TCGv tmp;



    switch (opsize) {

    case OS_BYTE:

        tmp = tcg_temp_new();

        if (sign)

            tcg_gen_ext8s_i32(tmp, val);

        else

            tcg_gen_ext8u_i32(tmp, val);

        break;

    case OS_WORD:

        tmp = tcg_temp_new();

        if (sign)

            tcg_gen_ext16s_i32(tmp, val);

        else

            tcg_gen_ext16u_i32(tmp, val);

        break;

    case OS_LONG:

    case OS_SINGLE:

        tmp = val;

        break;

    default:

        qemu_assert(0, "Bad operand size");

    }

    return tmp;

}
