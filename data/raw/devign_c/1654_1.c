av_cold void ff_fmt_convert_init_arm(FmtConvertContext *c, AVCodecContext *avctx)

{

    int cpu_flags = av_get_cpu_flags();



    if (have_vfp(cpu_flags)) {

        if (!have_vfpv3(cpu_flags)) {

            c->int32_to_float_fmul_scalar = ff_int32_to_float_fmul_scalar_vfp;

            c->int32_to_float_fmul_array8 = ff_int32_to_float_fmul_array8_vfp;

        }

    }



    if (have_neon(cpu_flags)) {

        c->int32_to_float_fmul_scalar = ff_int32_to_float_fmul_scalar_neon;

    }

}
