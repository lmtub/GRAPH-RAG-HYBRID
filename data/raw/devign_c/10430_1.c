static void shifter_out_im(TCGv var, int shift)

{

    TCGv tmp = new_tmp();

    if (shift == 0) {

        tcg_gen_andi_i32(tmp, var, 1);

    } else {

        tcg_gen_shri_i32(tmp, var, shift);

        if (shift != 31)

            tcg_gen_andi_i32(tmp, tmp, 1);

    }

    gen_set_CF(tmp);

    dead_tmp(tmp);

}
