void tcg_gen_atomic_cmpxchg_i32(TCGv_i32 retv, TCGv addr, TCGv_i32 cmpv,

                                TCGv_i32 newv, TCGArg idx, TCGMemOp memop)

{

    memop = tcg_canonicalize_memop(memop, 0, 0);



    if (!parallel_cpus) {

        TCGv_i32 t1 = tcg_temp_new_i32();

        TCGv_i32 t2 = tcg_temp_new_i32();



        tcg_gen_ext_i32(t2, cmpv, memop & MO_SIZE);



        tcg_gen_qemu_ld_i32(t1, addr, idx, memop & ~MO_SIGN);

        tcg_gen_movcond_i32(TCG_COND_EQ, t2, t1, t2, newv, t1);

        tcg_gen_qemu_st_i32(t2, addr, idx, memop);

        tcg_temp_free_i32(t2);



        if (memop & MO_SIGN) {

            tcg_gen_ext_i32(retv, t1, memop);

        } else {

            tcg_gen_mov_i32(retv, t1);

        }

        tcg_temp_free_i32(t1);

    } else {

        gen_atomic_cx_i32 gen;



        gen = table_cmpxchg[memop & (MO_SIZE | MO_BSWAP)];

        tcg_debug_assert(gen != NULL);



#ifdef CONFIG_SOFTMMU

        {

            TCGv_i32 oi = tcg_const_i32(make_memop_idx(memop & ~MO_SIGN, idx));

            gen(retv, tcg_ctx.tcg_env, addr, cmpv, newv, oi);

            tcg_temp_free_i32(oi);

        }

#else

        gen(retv, tcg_ctx.tcg_env, addr, cmpv, newv);

#endif



        if (memop & MO_SIGN) {

            tcg_gen_ext_i32(retv, retv, memop);

        }

    }

}
