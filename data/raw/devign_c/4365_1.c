static ExitStatus op_ex(DisasContext *s, DisasOps *o)

{

    /* ??? Perhaps a better way to implement EXECUTE is to set a bit in

       tb->flags, (ab)use the tb->cs_base field as the address of

       the template in memory, and grab 8 bits of tb->flags/cflags for

       the contents of the register.  We would then recognize all this

       in gen_intermediate_code_internal, generating code for exactly

       one instruction.  This new TB then gets executed normally.



       On the other hand, this seems to be mostly used for modifying

       MVC inside of memcpy, which needs a helper call anyway.  So

       perhaps this doesn't bear thinking about any further.  */



    TCGv_i64 tmp;



    update_psw_addr(s);

    update_cc_op(s);



    tmp = tcg_const_i64(s->next_pc);

    gen_helper_ex(cc_op, cpu_env, cc_op, o->in1, o->in2, tmp);

    tcg_temp_free_i64(tmp);



    set_cc_static(s);

    return NO_EXIT;

}
