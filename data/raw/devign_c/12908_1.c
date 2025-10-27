static void decode_opc(DisasContext * ctx)

{

    uint32_t old_flags = ctx->envflags;



    _decode_opc(ctx);



    if (old_flags & DELAY_SLOT_MASK) {

        /* go out of the delay slot */

        ctx->envflags &= ~DELAY_SLOT_MASK;

        tcg_gen_movi_i32(cpu_flags, ctx->envflags);

        ctx->bstate = BS_BRANCH;

        if (old_flags & DELAY_SLOT_CONDITIONAL) {

	    gen_delayed_conditional_jump(ctx);

        } else if (old_flags & DELAY_SLOT) {

            gen_jump(ctx);

	}



    }

}
