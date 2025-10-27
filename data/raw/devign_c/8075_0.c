static inline int tcg_temp_new_internal(TCGType type, int temp_local)

{

    TCGContext *s = &tcg_ctx;

    TCGTemp *ts;

    int idx, k;



    k = type + (temp_local ? TCG_TYPE_COUNT : 0);

    idx = find_first_bit(s->free_temps[k].l, TCG_MAX_TEMPS);

    if (idx < TCG_MAX_TEMPS) {

        /* There is already an available temp with the right type.  */

        clear_bit(idx, s->free_temps[k].l);



        ts = &s->temps[idx];

        ts->temp_allocated = 1;

        assert(ts->base_type == type);

        assert(ts->temp_local == temp_local);

    } else {

        idx = s->nb_temps;

#if TCG_TARGET_REG_BITS == 32

        if (type == TCG_TYPE_I64) {

            tcg_temp_alloc(s, s->nb_temps + 2);

            ts = &s->temps[s->nb_temps];

            ts->base_type = type;

            ts->type = TCG_TYPE_I32;

            ts->temp_allocated = 1;

            ts->temp_local = temp_local;

            ts->name = NULL;

            ts++;

            ts->base_type = TCG_TYPE_I32;

            ts->type = TCG_TYPE_I32;

            ts->temp_allocated = 1;

            ts->temp_local = temp_local;

            ts->name = NULL;

            s->nb_temps += 2;

        } else

#endif

        {

            tcg_temp_alloc(s, s->nb_temps + 1);

            ts = &s->temps[s->nb_temps];

            ts->base_type = type;

            ts->type = type;

            ts->temp_allocated = 1;

            ts->temp_local = temp_local;

            ts->name = NULL;

            s->nb_temps++;

        }

    }



#if defined(CONFIG_DEBUG_TCG)

    s->temps_in_use++;

#endif

    return idx;

}
