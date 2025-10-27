static inline void gen_op_fcmpeq(int fccno)

{

    switch (fccno) {

    case 0:

        gen_helper_fcmpeq(cpu_env);

        break;

    case 1:

        gen_helper_fcmpeq_fcc1(cpu_env);

        break;

    case 2:

        gen_helper_fcmpeq_fcc2(cpu_env);

        break;

    case 3:

        gen_helper_fcmpeq_fcc3(cpu_env);

        break;

    }

}
