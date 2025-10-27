static always_inline void check_cp1_3d(CPUState *env, DisasContext *ctx)

{

    if (unlikely(!(env->fpu->fcr0 & (1 << FCR0_3D))))

        generate_exception(ctx, EXCP_RI);

}
