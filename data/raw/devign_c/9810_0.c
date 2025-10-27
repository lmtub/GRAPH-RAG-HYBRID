static inline int check_fit_tl(tcg_target_long val, unsigned int bits)

{

    return (val << ((sizeof(tcg_target_long) * 8 - bits))

            >> (sizeof(tcg_target_long) * 8 - bits)) == val;

}
