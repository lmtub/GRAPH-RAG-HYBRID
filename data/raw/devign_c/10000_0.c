static inline uint32_t insn_get(CPUX86State *env, DisasContext *s, TCGMemOp ot)

{

    uint32_t ret;



    switch (ot) {

    case MO_8:

        ret = cpu_ldub_code(env, s->pc);

        s->pc++;

        break;

    case MO_16:

        ret = cpu_lduw_code(env, s->pc);

        s->pc += 2;

        break;

    case MO_32:

#ifdef TARGET_X86_64

    case MO_64:

#endif

        ret = cpu_ldl_code(env, s->pc);

        s->pc += 4;

        break;

    default:

        tcg_abort();

    }

    return ret;

}
