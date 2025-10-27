static inline void gen_op_fcmpes(int fccno, TCGv_i32 r_rs1, TCGv_i32 r_rs2)

{

    switch (fccno) {

    case 0:

        gen_helper_fcmpes(cpu_env, r_rs1, r_rs2);

        break;

    case 1:

        gen_helper_fcmpes_fcc1(cpu_env, r_rs1, r_rs2);

        break;

    case 2:

        gen_helper_fcmpes_fcc2(cpu_env, r_rs1, r_rs2);

        break;

    case 3:

        gen_helper_fcmpes_fcc3(cpu_env, r_rs1, r_rs2);

        break;

    }

}
