static int x86_cpu_filter_features(X86CPU *cpu)

{

    CPUX86State *env = &cpu->env;

    FeatureWord w;

    int rv = 0;



    for (w = 0; w < FEATURE_WORDS; w++) {

        uint32_t host_feat =

            x86_cpu_get_supported_feature_word(w, false);

        uint32_t requested_features = env->features[w];

        env->features[w] &= host_feat;

        cpu->filtered_features[w] = requested_features & ~env->features[w];

        if (cpu->filtered_features[w]) {

            rv = 1;

        }

    }



    return rv;

}
