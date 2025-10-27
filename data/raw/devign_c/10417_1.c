void HELPER(stfl)(CPUS390XState *env)

{

    uint64_t words[MAX_STFL_WORDS];



    do_stfle(env, words);

    cpu_stl_data(env, 200, words[0] >> 32);

}
