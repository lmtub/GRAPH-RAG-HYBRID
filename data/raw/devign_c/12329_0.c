void tcg_add_target_add_op_defs(const TCGTargetOpDef *tdefs)

{

    TCGOpcode op;

    TCGOpDef *def;

    const char *ct_str;

    int i, nb_args;



    for(;;) {

        if (tdefs->op == (TCGOpcode)-1)

            break;

        op = tdefs->op;

        assert((unsigned)op < NB_OPS);

        def = &tcg_op_defs[op];

#if defined(CONFIG_DEBUG_TCG)

        /* Duplicate entry in op definitions? */

        assert(!def->used);

        def->used = 1;

#endif

        nb_args = def->nb_iargs + def->nb_oargs;

        for(i = 0; i < nb_args; i++) {

            ct_str = tdefs->args_ct_str[i];

            /* Incomplete TCGTargetOpDef entry? */

            assert(ct_str != NULL);

            tcg_regset_clear(def->args_ct[i].u.regs);

            def->args_ct[i].ct = 0;

            if (ct_str[0] >= '0' && ct_str[0] <= '9') {

                int oarg;

                oarg = ct_str[0] - '0';

                assert(oarg < def->nb_oargs);

                assert(def->args_ct[oarg].ct & TCG_CT_REG);

                /* TCG_CT_ALIAS is for the output arguments. The input

                   argument is tagged with TCG_CT_IALIAS. */

                def->args_ct[i] = def->args_ct[oarg];

                def->args_ct[oarg].ct = TCG_CT_ALIAS;

                def->args_ct[oarg].alias_index = i;

                def->args_ct[i].ct |= TCG_CT_IALIAS;

                def->args_ct[i].alias_index = oarg;

            } else {

                for(;;) {

                    if (*ct_str == '\0')

                        break;

                    switch(*ct_str) {

                    case 'i':

                        def->args_ct[i].ct |= TCG_CT_CONST;

                        ct_str++;

                        break;

                    default:

                        if (target_parse_constraint(&def->args_ct[i], &ct_str) < 0) {

                            fprintf(stderr, "Invalid constraint '%s' for arg %d of operation '%s'\n",

                                    ct_str, i, def->name);

                            exit(1);

                        }

                    }

                }

            }

        }



        /* TCGTargetOpDef entry with too much information? */

        assert(i == TCG_MAX_OP_ARGS || tdefs->args_ct_str[i] == NULL);



        /* sort the constraints (XXX: this is just an heuristic) */

        sort_constraints(def, 0, def->nb_oargs);

        sort_constraints(def, def->nb_oargs, def->nb_iargs);



#if 0

        {

            int i;



            printf("%s: sorted=", def->name);

            for(i = 0; i < def->nb_oargs + def->nb_iargs; i++)

                printf(" %d", def->sorted_args[i]);

            printf("\n");

        }

#endif

        tdefs++;

    }



#if defined(CONFIG_DEBUG_TCG)

    i = 0;

    for (op = 0; op < tcg_op_defs_max; op++) {

        const TCGOpDef *def = &tcg_op_defs[op];

        if (def->flags & TCG_OPF_NOT_PRESENT) {

            /* Wrong entry in op definitions? */

            if (def->used) {

                fprintf(stderr, "Invalid op definition for %s\n", def->name);

                i = 1;

            }

        } else {

            /* Missing entry in op definitions? */

            if (!def->used) {

                fprintf(stderr, "Missing op definition for %s\n", def->name);

                i = 1;

            }

        }

    }

    if (i == 1) {

        tcg_abort();

    }

#endif

}
