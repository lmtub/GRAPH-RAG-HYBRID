int tcg_global_mem_new_internal(TCGType type, TCGv_ptr base,

                                intptr_t offset, const char *name)

{

    TCGContext *s = &tcg_ctx;

    TCGTemp *ts, *base_ts = &s->temps[GET_TCGV_PTR(base)];

    int idx, reg = base_ts->reg;



    idx = s->nb_globals;

#if TCG_TARGET_REG_BITS == 32

    if (type == TCG_TYPE_I64) {

        char buf[64];

        tcg_temp_alloc(s, s->nb_globals + 2);

        ts = &s->temps[s->nb_globals];

        ts->base_type = type;

        ts->type = TCG_TYPE_I32;

        ts->fixed_reg = 0;

        ts->mem_allocated = 1;

        ts->mem_reg = reg;

#ifdef HOST_WORDS_BIGENDIAN

        ts->mem_offset = offset + 4;

#else

        ts->mem_offset = offset;

#endif

        pstrcpy(buf, sizeof(buf), name);

        pstrcat(buf, sizeof(buf), "_0");

        ts->name = strdup(buf);

        ts++;



        ts->base_type = type;

        ts->type = TCG_TYPE_I32;

        ts->fixed_reg = 0;

        ts->mem_allocated = 1;

        ts->mem_reg = reg;

#ifdef HOST_WORDS_BIGENDIAN

        ts->mem_offset = offset;

#else

        ts->mem_offset = offset + 4;

#endif

        pstrcpy(buf, sizeof(buf), name);

        pstrcat(buf, sizeof(buf), "_1");

        ts->name = strdup(buf);



        s->nb_globals += 2;

    } else

#endif

    {

        tcg_temp_alloc(s, s->nb_globals + 1);

        ts = &s->temps[s->nb_globals];

        ts->base_type = type;

        ts->type = type;

        ts->fixed_reg = 0;

        ts->mem_allocated = 1;

        ts->mem_reg = reg;

        ts->mem_offset = offset;

        ts->name = name;

        s->nb_globals++;

    }

    return idx;

}
