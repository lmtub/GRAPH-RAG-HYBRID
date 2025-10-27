static unsigned do_stfle(CPUS390XState *env, uint64_t words[MAX_STFL_WORDS])

{

    S390CPU *cpu = s390_env_get_cpu(env);

    const unsigned long *features = cpu->model->features;

    unsigned max_bit = 0;

    S390Feat feat;



    memset(words, 0, sizeof(uint64_t) * MAX_STFL_WORDS);



    if (test_bit(S390_FEAT_ZARCH, features)) {

        /* z/Architecture is always active if around */

        words[0] = 1ull << (63 - 2);

    }



    for (feat = find_first_bit(features, S390_FEAT_MAX);

         feat < S390_FEAT_MAX;

         feat = find_next_bit(features, S390_FEAT_MAX, feat + 1)) {

        const S390FeatDef *def = s390_feat_def(feat);

        if (def->type == S390_FEAT_TYPE_STFL) {

            unsigned bit = def->bit;

            if (bit > max_bit) {

                max_bit = bit;

            }

            assert(bit / 64 < MAX_STFL_WORDS);

            words[bit / 64] |= 1ULL << (63 - bit % 64);

        }

    }



    return max_bit / 64;

}
