void HELPER(stfl)(CPUS390XState *env)

{

    uint64_t words[MAX_STFL_WORDS];

    LowCore *lowcore;



    lowcore = cpu_map_lowcore(env);

    do_stfle(env, words);

    lowcore->stfl_fac_list = cpu_to_be32(words[0] >> 32);

    cpu_unmap_lowcore(lowcore);

}
