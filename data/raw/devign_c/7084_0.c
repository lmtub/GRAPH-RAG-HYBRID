static int get_para_features(CPUState *env)

{

        int i, features = 0;



        for (i = 0; i < ARRAY_SIZE(para_features) - 1; i++) {

                if (kvm_check_extension(env->kvm_state, para_features[i].cap))

                        features |= (1 << para_features[i].feature);

        }



        return features;

}
