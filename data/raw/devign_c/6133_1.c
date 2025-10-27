static void dead_tmp(TCGv tmp)

{

    tcg_temp_free(tmp);

    num_temps--;

}
